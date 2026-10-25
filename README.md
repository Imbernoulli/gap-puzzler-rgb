# Puzzler GaP

Two simulated YAM arms build the five-layer IQ Puzzler Pro pyramid from twelve
loose pieces in RoboSimStudio (RSS). The controller is a Graph-as-Policy (GaP)
workflow: one typed subgraph per piece, independent physical checkpoints after
every node. It sees the table only through calibrated RGB cameras (four fixed cameras, or
the official YAM station rig: a gantry D405 and two wrist D405s), the robot's
joint encoders and the parts' CAD.

Site: <https://bohanlyu.com/gap-puzzler-rgb/>. Tabs switch between the official
three-camera rig (default) and our custom four-camera rig, which was built first.

## Results

Each run is a fresh native episode from twelve loose pieces (layout C: Lime starts 70 mm further out).
Success means all 84 independent checkpoints pass, the RSS goal holds and the GaP workflow reaches
its success exit.

| Code | Nominal runs | Complete | Notes |
| --- | --- | --- | --- |
| v13 (starting point) | 1 | 0 | All 12 seated (audit 2.53 mm); last node refused. |
| receipt recovery (b4cb667) | 2 | 0 | Cyan release; Red refused on a stale Purple flag. |
| `full12-success-v1` (7af3462) | 3 | 1 | `nominal_fix2c`: audit 2.55 mm, 41.8 s still. |
| fix3 (578a46f) | 3 | 1 | Two lost Cyan's release (75 mm opening sweep). |
| fix5 (7014075) | 4 | 3 | One stopped in Cyan's carry after the hand-to-hand transfer. |
| fix6 (81e08f7, Cyan transfer fixes) | 4 | 3 | One refused Mint's release on a 1.5-2 mm case shift (fixed next). |
| fix7 (a7491b9, + case-shift fix) | 4 | 4 | |
| fix8 (a1f87e0, + symmetric-alias fix) | 2 | 2 | |
| fix9 (5522892, + Maroon approach) | 4 | 3 | One stopped in Cyan's carry after the hand-to-hand transfer. |
| fix10 (3a06689, + Red transfer stations) | 3 | 3 | |
| fix11 (13c9010, + stricter completion witness) | 3 | 3 | |

Perturbed scenes, all fresh full runs from loose pieces: the current policy,
fix11 (13c9010), completed all seven (case moved and turned both ways, both
jittered layouts, friction 0.50, friction 0.80, all perturbations combined),
7/7; fix10 completed 6/6. The v13 policy completed none of them.

**Official camera rig** (gantry D405 + two wrist D405s, `experiments/official_rig/`):
the first complete run is `casefit_board_a2`, case moved +6/-6 mm and turned 2.9°:
12/12 pieces, 84/84 checkpoints, RSS and GaP success, audit 2.61 mm, 931 s simulated.
More official-rig scenes are still being run.

Perturbed scenes and scattered layouts are on the site.

## Layout

| Path | What it is |
| --- | --- |
| `policy/` | The policy workspace: `puzzler/` skills and perception, `graphs/` the GaP workflow, `verification/` the independent checkpoints, `profiles/` per-piece parameters. |
| `experiments/run.py` | Builds an isolated workspace from `policy/` and runs one fresh all-loose episode, optionally under a declared scene variant (case pose, piece jitter, friction, colours, scattered layouts). |
| `experiments/variant_native.py` | The RSS job for a variant. It changes only the initial scene; the policy never receives these numbers. |
| `tools/` | Renderers for the site videos, the completion-witness demo, the board-drift figure and the site generator. |
| `site_src/`, `site/` | Page sources and the built page that GitHub Pages serves. |

## Starting point and changes

`policy/` started as an exact copy of the v13 policy from the earlier
`gap-puzzler-web` work (first commit). v13 seated all twelve pieces
(independent audit 2.53 mm) but its workflow aborted at the last node. The
later commits change:

- **Receipts.** RGB contradictions of earlier placements are re-checked on later frames under the same thresholds; at the end, a receipt disputed by less than the 4 mm goal envelope stays valid and the dispute is recorded.
- **Completion.** `finish` asks the observer whether every bead site the cameras should see shows its part (`puzzler/assembly_completion.py`). Frames with a missing piece are refused.
- **Fit aliases.** A partly hidden piece fitted at a wrong turn no longer invalidates a seated piece when the goal pose explains the pixels.
- **Case re-look.** Release planning takes up to two more looks when the case fit contradicts the historical reference; the 1.5 mm limit is unchanged.
- **Purple release.** After an active held-piece inspection moves the hand, the held estimate is re-propagated at the current joint state.
- **Grasp search.** When a profile's tuned grasp is unreachable, every public bead and jaw axis is screened under the same checks.

## Running

```bash
python3 experiments/run.py prepare my_run --variant nominal
python3 experiments/run.py launch my_run --gpu 0 --cpu 0
python3 experiments/run.py status my_run
```

Workspaces link the pinned RSS and GaP checkouts from `gap-puzzler-web/third_party`
(`policy/RUNTIME.lock` lists the commits).
