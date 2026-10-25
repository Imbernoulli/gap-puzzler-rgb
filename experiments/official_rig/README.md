# Official camera rig

`policy.diff` turns `policy/` (commit 13c9010) into the policy that sees only the
official I2RT YAM station cameras: the gantry D405 (eye 0.0865, 0, 1.726, looking at
0.6497, 0, 0.75, 58° fov) and the two wrist D405s on `link_6`. It is the exact code of
round 29d (`officialcam29d_*`), which completed 4 of 6 fresh runs: nominal ×2 and all
perturbations combined ×2, each 12/12 with 84/84 checkpoints, RSS and GaP success and
an independent audit of 2.62-2.89 mm.

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

No checkpoint, gate or threshold is relaxed.

```bash
python3 experiments/run.py prepare off_run --variant board_a
(cd runs/off_run/workspace && git apply -p2 ../../../experiments/official_rig/policy.diff)
python3 experiments/run.py launch off_run --gpu 0 --cpu 0
```
