"""Deferred layout-C job authoring, with no environment construction.

Copy this module into the selected workspace's tools package only if the
layout-C policy qualifies. Its jobfile must call jobs() in both parent and
native worker preload processes. The original unified runner and policy are
retained; the scoped change affects only public initial scene compilation.
"""
import json
from pathlib import Path

from puzzler import model, task
from tools.unified_native import jobs as canonical_jobs

LAYOUT_NAME = "C_Lime_world_y_plus070"


def jobs(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    layout = manifest.get("public_initial_layout", {})
    if (layout.get("name") != LAYOUT_NAME
            or layout.get("piece") != "lime"
            or layout.get("translation_m") != [0., .070, 0.]
            or layout.get("all12_loose") is not True
            or manifest.get("canonical_original_staging") is not False
            or manifest.get("initialized_prefix") != []
            or manifest.get("initial_ledgers_empty") is not True):
        raise ValueError("An explicit fresh all-loose layout-C manifest is required")

    original_staging = task.staging

    def public_staging_c():
        stage = {name: (center.copy(), yaw, side)
                 for name, (center, yaw, side) in original_staging().items()}
        if set(stage) != set(model.CELLS):
            raise ValueError("Layout C requires all twelve public loose pieces")
        center, yaw, side = stage["lime"]
        center[1] += .070
        stage["lime"] = (center, yaw, side)
        return stage

    try:
        task.staging = public_staging_c
        # This calls the original task_spec, registering the shifted body
        # assets before returning the unmodified UnifiedPyramidGraph JobSpec.
        return canonical_jobs(manifest_path)
    finally:
        task.staging = original_staging
