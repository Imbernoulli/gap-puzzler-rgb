# Puzzler GaP

Two simulated YAM arms build the five-layer IQ Puzzler Pro pyramid from twelve
loose pieces in RoboSimStudio (RSS). The controller is a Graph-as-Policy (GaP)
workflow: one typed subgraph per piece, independent physical checkpoints after
every node. It sees the table only through four calibrated RGB cameras, the
robot's joint encoders and the parts' CAD.

Site: <https://bohanlyu.com/gap-puzzler-rgb/>

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
