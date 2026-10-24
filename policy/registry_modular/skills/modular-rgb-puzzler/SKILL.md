---
name: modular-rgb-puzzler
description: Controls known puzzle pieces through actual RGB perception, measured robot state and public geometry skill modules. Use when a source grasp and transport/release plan must be recomputed from the current scene rather than replayed from demonstration commands.
metadata: {category: robotics, tags: [gpu, rgb, geometry]}
gap:
  requires:
    gpu: true
    connector: [vision.observe, puzzle.plan_pose, puzzle.plan_path, puzzle.plan_joint_path, puzzle.plan_pose_sequence, puzzle.propagate_pyramid_held_belief, puzzle.capture_conditional_attachment, puzzle.propagate_conditional_attachment, robot.execution_journal_cursor, robot.read_proprioception, robot.grasp_frame, robot.set_grip, robot.wait_steps]
  allowed_tools: [vision.observe, puzzle.plan_pose, puzzle.plan_path, puzzle.plan_joint_path, puzzle.plan_pose_sequence, puzzle.propagate_pyramid_held_belief, puzzle.capture_conditional_attachment, puzzle.propagate_conditional_attachment, robot.execution_journal_cursor, robot.read_proprioception, robot.get_ee_pose, robot.grasp_frame, robot.set_grip, robot.execute_trajectory, robot.wait_steps, robot.go_home]
  required_inputs: {state: ModularState}
  produces_outputs: {state: ModularState, evidence: dict, decision: str}
  exit_conditions:
    planned: Current measured source andexplicit current/historical scene beliefs support recomputed grasp/IK/CADplan.
    verified: Fresh heldRGB source or explicit conditionalattachment drivescontrol; independent physicalCP must verify realheld target. UNKNOWN is notmeasuredheldpose.
    receiver_ready: Cyan donor retains the piece while the receiver is open at a freshly screened approach; independent physical approach checkpoint required.
    receiver_closed: Cyan receiver has actually closed; separate continuous dual hold physical checkpoint required before donor opens.
    donor_released: Cyan donor opened and withdrew; independent receiver-only physical checkpoint required.
    receiver_verified: Actual receiver owns a new RGB binding or an explicitly conditional dated RGB and actual-close q16 binding; independent retention checkpoint required.
    prepared: Red donor remains physically held and receiver remains open at the screened approach; the declared hop has a separate independent physical checkpoint.
    dual_held: Both named hands actually hold Red continuously; a separate physical checkpoint forbids every other support before donor opens.
    transferred: Red donor actually opened and withdrew while the actual receiver alone retains the piece; independent receiver-only physical checkpoint required.
    carried: Currentrobot feedback reaches dynamically recomputed publicCADtarget hover; independent verifier checksheld object.
    placed: Actual releasepair supports a stable3mm visual placement; independent verifier still checksphysical placement.
    uncertain: Release completed but visualplacement proof remains uncertain; independent verifier must provephysical placement before anyhandoff.
    unknown: Bounded current RGB/CAD reobservation did not produce an authentic placement receipt; earned remains unchanged. Only an explicitly configured conditional execution route may hand off after the original independent physical checks pass.
    executed_visual_unknown: A configured intermediate piece or final Red completed release with actual execution and RGB provenance but no visual receipt; preserve the explicit conditional hypothesis, leave earned unchanged, and require independent physical verification before continuing.
    executed_visual_unverified: A configured intermediate piece or final Red has an actual visible but unverified or conflicting RGB pair; preserve its measured error without minting a receipt and require independent physical verification for the configured conditional route.
    missed: Two real released RGB frames visibly miss the goal; preserve the raw miss and conflict. Abort, explicitly recover, or use an explicitly configured conditional execution route only after its independent physical checks pass; never automatically label it placed.
    advanced: Only actualpreceding release execution movesstate to nextpiece; UNKNOWNis not upgraded toplaced.
    completed: All twelve actual execution tokens have authentic visual receipts or explicitly permitted conditional execution hypotheses; canonical full12 has no initialized prefix. Both hands are open and home for at least two seconds and independent all12 physical completion passes; visual UNKNOWN and raw conflicts remain explicit.
    error: Evidence, geometry orphysical postcondition refused.
  streaming: false
---

Source and release openingwidths are independent. Public grasp profiles contain only CAD-relative bead/heading/lean/H and skill parameters. All worldSE3, robot IK paths andclearance checks are recomputed from RGB/proprio/publicCAD. Unknown observations are not filled fromgoals. Independent test verifier truth stays outside policyWorld/tools/planner.

Native simulation and tiled RGB rendering require an explicitly selected CUDA GPU. Use the delivery GPU launcher; it checks CUDA availability before starting the graph.
