"""Tile several recorded episodes into one time-aligned N×N video.

  python tools/make_grid.py OUT.mp4 --gpu N --runs runs/a runs/b ... [--stride 6]

Every cell starts at its own t = 0 and holds its last frame after its episode
ends. The label shows the scene and how many pieces that episode placed.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path('/srv/home/bohanlyu/gap-puzzler-web')
RUN = 'full12_original_order_layout_c_v10'
CELL = (426, 240)


def handoffs(run):
    log = run / 'independent_verification/checkpoint_results.jsonl'
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return [r['native_step'] for r in rows if r.get('phase') == 'handoff' and r.get('passed')]


def main():
    import imageio.v2 as iio
    from PIL import Image, ImageDraw, ImageFont
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--stride', type=int, default=6)
    parser.add_argument('--pieces', type=int, default=0, help='end each clip at its Nth placed piece')
    parser.add_argument('--labels', nargs='*', default=[], help='cell captions, in --runs order')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    cells_dir = out.with_suffix('')
    cells_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for k, path in enumerate(map(Path, args.runs)):
        run = path.resolve() / 'workspace/runs' / RUN
        manifest = json.loads((run / 'manifest.json').read_text())
        steps = handoffs(run)
        clip = cells_dir / f'{path.name}_{args.pieces}.mp4'
        extra = ['--end-step', str(steps[args.pieces - 1] + 100)] if args.pieces and len(steps) >= args.pieces else []
        if not clip.exists():
            env = dict(__import__('os').environ, RSS_WORKSPACE=str(run.parents[1]))
            subprocess.run([str(UPSTREAM / 'tools/rss_gpu.sh'), str(args.gpu), str(ROOT / 'tools/render_gap.py'),
                            str(run / 'episode'), str(clip), '--stride', str(args.stride), '--plain', *extra],
                           cwd=run.parents[1], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        frames = [np.asarray(Image.fromarray(f).resize(CELL, Image.LANCZOS)) for f in iio.mimread(str(clip), memtest=False)]
        shown = min(len(steps), args.pieces) if args.pieces else len(steps)
        cells.append(dict(frames=frames, label=args.labels[k] if k < len(args.labels) else manifest.get('variant_label', path.name),
                          status='done ✓' if shown == 12 else f"{shown} placed"))
    side = math.ceil(math.sqrt(len(cells)))
    length = max(len(c['frames']) for c in cells)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except OSError:
        font = ImageFont.load_default()
    with iio.get_writer(str(out), fps=30, codec='libx264', quality=7, output_params=['-pix_fmt', 'yuv420p']) as writer:
        for i in range(length):
            canvas = Image.new('RGB', (side * CELL[0], side * CELL[1]), (20, 22, 26))
            for k, cell in enumerate(cells):
                frame = Image.fromarray(cell['frames'][min(i, len(cell['frames']) - 1)])
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, CELL[0], 22), fill=(21, 23, 26))
                draw.text((8, 3), cell['label'], fill=(240, 240, 240), font=font)
                if i >= len(cell['frames']) - 1:
                    draw.text((CELL[0] - draw.textlength(cell['status'], font=font) - 8, 3), cell['status'], fill=(140, 220, 160), font=font)
                canvas.paste(frame, ((k % side) * CELL[0], (k // side) * CELL[1]))
            writer.append_data(np.asarray(canvas))
    print(out, length, 'frames')


if __name__ == '__main__':
    main()
