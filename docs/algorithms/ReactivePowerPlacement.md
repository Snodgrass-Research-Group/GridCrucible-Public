# ReactivePowerPlacement.m + PVPQSwitching.m → `network/shunts.py`

**Sources:** ReactivePowerPlacement.m (235 loc), PVPQSwitching.m (47 loc)
Pipeline stage 7 (after create_buses).

## The algorithm: synchronous-condenser winnowing

Rather than deciding where shunt compensation is needed, MATLAB starts with
support EVERYWHERE and removes it where it isn't used:

1. **Prep**: cap line charging b_pu at 1; power-factor-correct the load Q
   (targets by band: PF ≤ 0.9 → 0.97, ≤ 0.95 → 0.98, ≤ 0.96 → 0.99);
   take the max ED scenario (first block) from the Complete tier and
   duplicate it as scenario 1; loads/committed gens from that scenario.
2. **Init**: slack from the bus-level roles; EVERY other bus becomes a PV
   bus at 1.00 pu — i.e. a synchronous-condenser CANDIDATE with shunt
   limits HighQ = 800 MVar (withdrawal) / LowQ = −1200 MVar (injection).
3. **Winnowing loop**:
   * demote the `per_switch` = 5% of candidates with the SMALLEST |Q|
     support to plain PQ buses (they don't need compensation);
   * re-solve the AC power flow (PV setpoints re-normalized to 1 pu);
   * non-convergence ⇒ revert the demotion and shrink per_switch ×0.9
     (give up below 1%);
   * buses exceeding the shunt Q limits or voltage-triggered at the limits
     get the PV↔PQ treatment (PVPQSwitching.m — two fixed re-solves
     clamping Q at HighQ/LowQ). **Ported explicitly** (2026-07-11): at
     full scale, pandapower's enforce_q_lims PV↔PQ switching collapses
     weak buses to V=0 (exactly-singular Jacobian) — the original
     "natively subsumed" shortcut only holds on small cases. `_solve`
     runs the NR→Gauss-Seidel ladder (power_flow.robust_runpp — the
     synthetic network's 6-decade impedance span leaves the NR Jacobian
     numerically singular, σ_min ≈ 3e-13 at a converged point; GS reaches
     1 MVA ≈ 2e-5 relative, and MATLAB's ACPowerFlowFinal recorded
     PF.Converge without enforcing it) and then applies MATLAB's two
     clamping passes verbatim;
   * buses with |V| outside [0.94, 1.06] receive PERMANENT compensation
     (`sync_cond` = True, stay PV).
4. **Terminate** when remaining candidates + placed < 20% of buses: all
   remaining candidates become permanent condensers. Slack reasserted at
   the max-capacity bus.

Outputs: `buses.sync_cond` (where shunt compensation lives), `buses.v_set`,
bus types, and the final converged ACPFResult (feeds TUNELoop.run_ac).

## pandapower mapping

Candidates = `gen` elements with p_mw = 0, vm_pu = 1.0,
min_q_mvar = −HighQ, max_q_mvar = −LowQ (sign flip: MATLAB QNet is
load-positive); demotion = drop the element; `enforce_q_lims=True` performs
PVPQSwitching's clamping inside each solve. Real generators keep their unit
Q limits and dispatch from the scenario.

## Open questions

1. HighQ/LowQ (800/−1200 MVar) are hard-coded bank sizes — config knobs in
   the port ([tune] section? currently module constants mirroring MATLAB).
2. The MATLAB loop's second/third re-solve blocks are near-duplicates
   (copy-paste evolution) — ported as ONE loop body per iteration;
   behavior-equivalent since each block re-solves from the last converged
   state.
3. `maxscenario` scans only the first 42 columns ("economic dispatch
   scenarios") — 42 is a magic number ≈ ED block width in the reference
   run; ported as the ED block width from the tier metadata.
