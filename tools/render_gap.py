"""External-camera replay of a native GaP episode from its recorded body poses.

Usage (from the episode's workspace):
  RSS_WORKSPACE=<ws> tools/rss_gpu.sh GPU render_gap.py EPISODE_DIR OUT.mp4 [--stride N] [--frames N]

Nothing is re-simulated: each frame assigns recorded body_q/joint_q and renders.
The caption reads the graph's own checkpoint log for the active piece/phase.
This camera is for viewers only; it is not one of the policy's four RGB inputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

PHASE = {'perceive_plan': 'perceive and plan', 'grasp': 'grasp and lift', 'transport': 'carry',
         'release': 'release', 'conditional_release': 'release', 'receipt_settle': 'check placement',
         'handoff': 'hand off to next piece', 'finish': 'both arms home',
         'handoff_approach': 'bring hands together', 'handoff_prepared': 'bring hands together',
         'handoff_dualhold': 'both hands hold', 'handoff_receiveronly': 'first hand lets go',
         'handoff_receiver_only': 'first hand lets go', 'handoff_receiver_hold': 'second hand holds',
         'handoff_receiver_bound': 'second hand holds'}


def phase_label(phase):
    if phase.startswith('return_'):
        return 'transfer back: ' + PHASE.get(phase[7:], phase[7:])
    return PHASE.get(phase, phase.replace('_', ' '))


CAMERA = ((1.12, -.10, 1.42), (.45, 0., .77))


def render(episode, out, stride, limit, start=0, end=None, camera=None, plain=False):
    import robosimstudio as rss
    from puzzler.task import scene, Variant
    import imageio.v2 as iio
    from PIL import Image, ImageDraw, ImageFont
    episode = Path(episode)
    result = json.loads((episode / 'result.json').read_text())
    variant = dict(result['variant'])
    variant['preset_pieces'] = tuple(variant.get('preset_pieces', ()))
    sim = scene(Variant(**variant)).build()
    eye, look = camera or CAMERA
    cam = rss.CameraSpec(name='replay', width=1280, height=720, fov_y=44, eye=eye, look_at=look, up=(0, 0, 1))
    frames = json.loads((episode / 'frames.json').read_text())
    frames = [f for f in frames if f['step'] >= start and (end is None or f['step'] <= end)]
    picked = frames[::stride]
    if picked[-1]['step'] != frames[-1]['step']:
        picked.append(frames[-1])
    if limit:
        picked = picked[:limit]
    log = episode.parent / 'independent_verification/checkpoint_results.jsonl'
    checks = [json.loads(line) for line in log.read_text().splitlines()]
    checks = sorted((c for c in checks if c.get('native_step') is not None), key=lambda c: c['native_step'])
    hz = result['control_hz']
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 20)
    except OSError:
        font = ImageFont.load_default()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with iio.get_writer(str(out), fps=30, codec='libx264', quality=7,
                        output_params=['-threads', '2', '-pix_fmt', 'yuv420p']) as writer:
        for i, frame in enumerate(picked):
            sim.state.body_q.assign(np.asarray(frame['body_q'], np.float32))
            sim.state.joint_q.assign(np.asarray(frame['joint_q'], np.float32))
            image = Image.fromarray(sim.render(cam)['rgb'])
            done = [c for c in checks if c['native_step'] <= frame['step']]
            placed = sum(1 for c in done if c.get('phase') == 'handoff' and c.get('passed'))
            active = next((c for c in checks if c['native_step'] > frame['step']), None)
            text = f"t = {frame['step'] / hz:6.1f} s   placed {min(placed, 12):2d}/12"
            if active is not None:
                text += f"   {active.get('piece') or 'all'}: {phase_label(active['phase'])}"
            if not plain:
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1280, 40), fill=(23, 60, 49))
                draw.text((18, 9), text, fill=(243, 244, 233), font=font)
            writer.append_data(np.asarray(image))
            if i == 0:
                image.save(out.with_name(out.stem + '_first.png'))
        image.save(out.with_name(out.stem + '_final.png'))
    print(out, len(picked), 'frames')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('episode')
    parser.add_argument('out')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--start-step', type=int, default=0)
    parser.add_argument('--end-step', type=int)
    parser.add_argument('--camera', help='eye_x,eye_y,eye_z,look_x,look_y,look_z')
    parser.add_argument('--plain', action='store_true', help='no caption bar')
    args = parser.parse_args()
    camera = None
    if args.camera:
        v = [float(x) for x in args.camera.split(',')]
        camera = (tuple(v[:3]), tuple(v[3:]))
    render(args.episode, args.out, args.stride, args.frames, args.start_step, args.end_step, camera, args.plain)
