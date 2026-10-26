# Official camera rig

`policy.diff` turns `policy/` (commit 13c9010) into the policy that sees only the
official I2RT YAM station cameras: the gantry D405 (eye 0.0865, 0, 1.726, looking at
0.6497, 0, 0.75, 58° fov) and the two wrist D405s on `link_6`. It is the exact code of
round 30 (`officialcam30_*`), which completed 5 of 6 fresh runs (nominal, combined, case
moved and turned the other way, second jittered layout, and a different starting layout),
each 12/12 with 84/84 checkpoints, RSS and GaP success and an independent audit of
2.65-2.70 mm.

Main changes:

- **Free-hand looks.** While one hand holds a part, the other moves its wrist camera to a
  screened look pose, and all three cameras are read together (`free_hand_look.py`, `survey.py`).
- **Case fit.** One fit of the case CAD outline across every view that sees it, with the
  robot's own silhouette masked out (`board_vision.fit_board_views`).
- **Held-part alias.** When the frame allows several shared-bead poses of a held part, keep
  the one within 6 mm of the encoder-propagated prediction if every other candidate is at
  least twice as far; otherwise abstain (`vision_observer._disambiguate_held`).
- **Occluded goal sites.** The goal-pixel test skips sites hidden by a held part or by the
  robot (`assembly_completion.goal_support`).
- **Finish.** The completion check pools the home frame with wrist looks that cover pieces
  the gantry cannot see.
- **Screened free-arm motion.** Every look move of the free arm is checked against both arms,
  the table, the case and every known piece before it runs; a hand bracing the case first
  leaves straight up (`free_arm_motion.py`).
- **Held pieces re-measured.** A piece held long enough to have turned in the grip is measured
  again from RGB at the transport hover, or by the free wrist, before it is placed.
- **Messy starts.** Loose pieces no camera sees are looked for with screened wrist looks;
  pieces whose grasp cannot be planned where they lie are moved or turned on the table first.
- **Shape, not colour.** Pieces are named by matching their CAD shape; each piece's colour is
  measured from its own beads and used only after that.
- **Apex reach.** Before the last hand-off, the top piece's final grasp is checked for reach
  at the measured case; if it fails, the same pinch rolled 10° or 20° is tried
  (`red_bimanual_handoff.py`).

No checkpoint, gate or threshold is relaxed.

Apply it with `patch`, not `git apply`: the workspace sits inside this repository, so
`git apply` would resolve paths against the repository root and apply nothing.

```bash
python3 experiments/run.py prepare off_run --variant board_a
(cd runs/off_run/workspace && patch -p2 < ../../../experiments/official_rig/policy.diff)
python3 experiments/run.py launch off_run --gpu 0 --cpu 0
```
