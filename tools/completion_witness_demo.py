"""Run the policy's completion witness on two recorded frames and draw it.

  python tools/completion_witness_demo.py EPISODE_DIR MISSING_INDEX COMPLETE_INDEX OUT_JPG OUT_JSON

Inputs are the policy's own saved RGB frames and its own RGB case estimate for
each frame; green marks a bead site that shows its part, red one that doesn't.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from puzzler import assembly_completion
from puzzler.vision import calibration, rig_specs
from puzzler.vision_observer import goals

COLORS = {'match': (40, 200, 90), 'mismatch': (235, 60, 40), 'unclear': (170, 170, 170)}


def main():
    episode, missing, complete, out_jpg, out_json = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
    observations = json.loads((episode / 'rgb/observations.json').read_text())
    cameras = calibration(rig_specs())
    targets = goals('pyramid', 'pyramid_robot6_plan.json')
    panels, summary = [], {}
    for label, index in (('missing', missing), ('complete', complete)):
        images = dict(np.load(episode / f'rgb/{index:04d}.npz'))
        result = assembly_completion.evaluate(images, cameras, observations[index]['poses']['board'], targets)
        summary[label] = dict(index=index, timestamp=observations[index]['timestamp'], complete=result['complete'],
                              decisive=result['decisive_views'], mismatched=result['mismatched_views'],
                              matched=result['decisive_views'] - result['mismatched_views'],
                              unconfirmed=result['unconfirmed_pieces'])
        for camera in ('top', 'front'):
            image = Image.fromarray(images[camera])
            draw = ImageDraw.Draw(image)
            views = [v for v in result['views'] if v['camera'] == camera]
            for view in views:
                u, v = view['pixel']
                draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=COLORS[view['outcome']], width=3)
            cx, cy = np.mean([v['pixel'] for v in views], axis=0)
            half = 150 if camera == 'top' else 130
            panels.append(image.crop((int(cx - 1.25 * half), int(cy - half), int(cx + 1.25 * half), int(cy + half))).resize((375, 300)))
    sheet = Image.new('RGB', (375 * 4 + 30, 300), (246, 247, 248))
    for k, panel in enumerate(panels):
        sheet.paste(panel, (k * 385, 0))
    sheet.save(out_jpg, quality=86)
    Path(out_json).write_text(json.dumps(summary, indent=1) + '\n')
    print(summary)


if __name__ == '__main__':
    main()
