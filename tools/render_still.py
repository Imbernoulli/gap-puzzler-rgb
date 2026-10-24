"""Render one recorded frame of an episode from several viewer cameras.

  render_still.py EPISODE_DIR OUT_DIR STEP name:ex,ey,ez,lx,ly,lz[,fov] ...
"""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    import robosimstudio as rss
    from puzzler.task import scene, Variant
    from PIL import Image
    episode, out, step = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    result = json.loads((episode / 'result.json').read_text())
    variant = dict(result['variant'])
    variant['preset_pieces'] = tuple(variant.get('preset_pieces', ()))
    sim = scene(Variant(**variant)).build()
    frames = json.loads((episode / 'frames.json').read_text())
    frame = min(frames, key=lambda f: abs(f['step'] - step)) if step >= 0 else frames[-1]
    sim.state.body_q.assign(np.asarray(frame['body_q'], np.float32))
    sim.state.joint_q.assign(np.asarray(frame['joint_q'], np.float32))
    for spec in sys.argv[4:]:
        name, values = spec.split(':')
        v = [float(x) for x in values.split(',')]
        cam = rss.CameraSpec(name=name, width=1280, height=720, fov_y=v[6] if len(v) > 6 else 40,
                             eye=tuple(v[:3]), look_at=tuple(v[3:6]), up=(0, 0, 1))
        Image.fromarray(sim.render(cam)['rgb']).save(out / f'{name}.png')
    print(out, frame['step'])


if __name__ == '__main__':
    main()
