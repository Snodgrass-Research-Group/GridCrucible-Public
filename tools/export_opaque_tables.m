%% export_opaque_tables.m — one-time MATLAB export for GridCrucible (v2)
% scipy cannot read MATLAB *table* objects (they save as opaque blobs), so
% run THIS script in MATLAB once, then re-run the Python migration
% (python -m gridcrucible.io.matlab_import) — it picks the CSVs up from
% gridcrucible/data/matlab_exports/ automatically.
%
% v2 (2026-07-08 audit of all .mat under Conductor Data Tables /
% Default Input Data / SRTV): adds Form715LimitFits, PJM/WECC gen data, and
% the OPTIONAL reference-run exports from SRTV/Example Datasets (regression
% targets for validating the Python conversion). Everything is wrapped in
% try/catch — partial success is fine, just re-run the Python migration.
%
% Run from the repo root (Synthetic-Grid-Generator/).

kvl    = fullfile(pwd, 'KCL-KVL Compliant Grid Generator');
srcdir = fullfile(kvl, 'Default Input Data');
netdir = fullfile(srcdir, 'Network Data');
exdir  = fullfile(kvl, 'SRTV', 'Example Datasets');
outdir = fullfile(pwd, 'gridcrucible', 'data', 'matlab_exports');
refdir = fullfile(outdir, 'reference_run');
if ~exist(outdir, 'dir'); mkdir(outdir); end
if ~exist(refdir, 'dir'); mkdir(refdir); end

function try_export(matfile, varname, csvout)
    try
        S = load(matfile);
        T = S.(varname);
        writetable(T, csvout);
        fprintf('exported %s -> %s\n', varname, csvout);
    catch err
        fprintf('SKIPPED %s (%s): %s\n', matfile, varname, err.message);
    end
end

%% Pipeline inputs (required)
try_export(fullfile(srcdir, 'XFData.mat'), 'XFData', fullfile(outdir, 'XFData.csv'));
try_export(fullfile(srcdir, 'XFXR.mat'),  'XFXR',  fullfile(outdir, 'XFXR.csv'));
% Form715LimitFits (v4): the variables are prob.NormalDistribution objects
% (fitted per-kV MVA-limit distributions, e.g. Fit138Form715) — export
% their parameters (mu/sigma) to ONE params CSV; Python reconstructs them
% as scipy.stats.norm.
try
    S = load(fullfile(netdir, 'Form715LimitFits.mat'));
    fn = fieldnames(S);
    rows = {};
    for i = 1:numel(fn)
        v = S.(fn{i});
        if istable(v)
            writetable(v, fullfile(outdir, ['Form715LimitFits_' fn{i} '.csv']));
            fprintf('exported Form715LimitFits.%s (table)\n', fn{i});
        elseif isa(v, 'prob.ProbabilityDistribution')
            pn = v.ParameterNames; pv = v.ParameterValues;
            for j = 1:numel(pn)
                rows(end+1,:) = {fn{i}, class(v), pn{j}, pv(j)}; %#ok<AGROW>
            end
            fprintf('captured Form715LimitFits.%s (%s)\n', fn{i}, class(v));
        else
            fprintf('Form715LimitFits.%s is a %s - not exported\n', ...
                fn{i}, class(v));
        end
    end
    if ~isempty(rows)
        T = cell2table(rows, 'VariableNames', {'fit','dist','param','value'});
        writetable(T, fullfile(outdir, 'Form715LimitFits_params.csv'));
        fprintf('exported Form715LimitFits_params.csv (%d rows)\n', height(T));
    end
catch err
    fprintf('SKIPPED Form715LimitFits: %s\n', err.message);
end
try_export(fullfile(netdir, 'PJMGenData.mat'),  'PJMGenData', ...
           fullfile(outdir, 'PJMGenData.csv'));
try_export(fullfile(netdir, 'WECCGenData.mat'), 'WECCGenData', ...
           fullfile(outdir, 'WECCGenData.csv'));
% NOTE: variable names inside the .mat may differ from the file stem — if a
% SKIPPED line says so, run `whos('-file', <matfile>)` and adjust varname.

%% Reference-run exports (OPTIONAL — regression targets, Example Datasets)
try_export(fullfile(kvl, 'SRTV', 'Gen.mat'), 'Gen', fullfile(outdir, 'Gen_reference.csv'));
try_export(fullfile(exdir, 'GenBuses.mat'), 'Gen', fullfile(refdir, 'GenBuses.csv'));
try
    S = load(fullfile(exdir, 'QNCAResults.mat'));
    writetable(S.Bus,      fullfile(refdir, 'QNCAResults_Bus.csv'));
    writetable(S.Sub,      fullfile(refdir, 'QNCAResults_Sub.csv'));
    writetable(S.Branch,   fullfile(refdir, 'QNCAResults_Branch.csv'));
    writetable(S.OLBranch, fullfile(refdir, 'QNCAResults_OLBranch.csv'));
    fprintf('exported QNCAResults tables\n');
catch err
    fprintf('SKIPPED QNCAResults: %s\n', err.message);
end
try
    S = load(fullfile(exdir, 'PFDataMetrics.mat'));
    writetable(S.PF.DCStats,    fullfile(refdir, 'PF_DCStats.csv'));
    writetable(S.PF.DCOPFStats, fullfile(refdir, 'PF_DCOPFStats.csv'));
    fprintf('exported PFDataMetrics stats\n');
catch err
    fprintf('SKIPPED PFDataMetrics: %s\n', err.message);
end
try
    S = load(fullfile(exdir, 'ACPF.mat'));
    writetable(S.ACPF.Stats, fullfile(refdir, 'ACPF_Stats.csv'));
    fprintf('exported ACPF stats\n');
catch err
    fprintf('SKIPPED ACPF: %s\n', err.message);
end

%% Load-profile bridge exports (OPTIONAL but recommended — see
%% docs/algorithms/MakeLoadProfiles.md "Port strategy"): run each region's
%% MakeLoadProfiles_<R> and save P/Q as plain -v6 doubles (scipy-readable).
%% Needs the Optimization Toolbox (quadprog inside smoothprofile).
regions = {'SRTV'};   % add regions here as their runs are needed
mlddir = fullfile(kvl, 'Master Load Data');
for i = 1:numel(regions)
    r = regions{i};
    try
        lp = fullfile(mlddir, r);
        addpath(genpath(lp));
        [P_Load, Q_Load] = feval(['MakeLoadProfiles_' r], lp);
        save(fullfile(outdir, [r '_LoadProfiles.mat']), ...
             'P_Load', 'Q_Load', '-v6');
        rmpath(genpath(lp));
        fprintf('exported %s load profiles (%dx%d)\n', r, size(P_Load,1), size(P_Load,2));
    catch err
        fprintf('SKIPPED %s load profiles: %s\n', r, err.message);
    end
end

fprintf('Done. Outputs in %s\n', outdir);
