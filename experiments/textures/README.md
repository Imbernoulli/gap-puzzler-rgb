# Wooden pieces

The site's "Wooden pieces" cell is `tex_woodspecies_nominal3`: the official-rig policy
(`experiments/official_rig/policy.diff` as of round 33, md5 9d437f38) plus two deltas,
applied inside a prepared workspace with `patch -p1`:

- `wood_species_scene.diff` (scene only, `puzzler/task.py`): each piece is a real wood
  (maple, zebrano, yellowheart, olive, amber oak, teak, cherry, padauk, bubinga, sapele,
  mahogany, walnut) with its grain as a texture. The policy is never told the appearance.
- `wood_species_perception.diff` (`appearance.py`, `shape_vision.py`, `red_rgb_recovery.py`):
  the fit outline also accepts dim pixels of a piece's own colour inside its own footprint,
  so a wood's shaded side is not lost. No gate or threshold changes.

```bash
python3 experiments/run.py prepare wood_run --variant wood_species
(cd runs/wood_run/workspace && patch -p2 < ../../../experiments/official_rig/policy.diff \
  && patch -p1 < ../../../experiments/textures/wood_species_scene.diff \
  && patch -p1 < ../../../experiments/textures/wood_species_perception.diff)
```

That run built all 12 pieces with every checkpoint, the RSS goal and the GaP success exit;
the worst bead ended 2.65 mm from its goal.
