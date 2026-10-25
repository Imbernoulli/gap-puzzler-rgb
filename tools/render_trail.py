"""Replay one piece's hand-to-hand transfer with its path and orientation drawn in.

  render_trail.py EPISODE_DIR OUT.mp4 PIECE START_STEP END_STEP ex,ey,ez,lx,ly,lz [--stride N] [--label TEXT]

Run inside the run's workspace (RSS_WORKSPACE). Nothing is re-simulated: each
frame assigns the recorded body state and renders it; the overlay is the
piece's recorded pose projected into the same camera. The piece's beads are
ringed; a dial (a disc in the piece's own plane, with a needle along its first
axis and a pin along its normal) shows its orientation; the centroid path and
faded copies of the dial stay on screen so the turn in 3D can be read.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

W, H, FOV = 1280, 720, 44


def look_matrix(eye, look_at, up=(0., 0., 1.)):
    eye, look_at, up = (np.asarray(v, float) for v in (eye, look_at, up))
    f = look_at - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, np.cross(f, r), f, eye
    return m


def project(points, inv):
    p = (inv[:3, :3] @ np.atleast_2d(points).T + inv[:3, 3:4]).T
    fpx = .5 * H / np.tan(np.radians(FOV) / 2)
    return np.c_[W / 2 + fpx * p[:, 0] / p[:, 2], H / 2 + fpx * p[:, 1] / p[:, 2]]


def dial(center, rot, radius=.040):
    t = np.linspace(0, 2 * np.pi, 49)
    x, y, z = rot[:, 0], rot[:, 1], rot[:, 2]
    ring = center + radius * (np.cos(t)[:, None] * x + np.sin(t)[:, None] * y)
    return ring, center + radius * x, center + .8 * radius * z


def main():
    import imageio.v2 as iio
    import robosimstudio as rss
    from PIL import Image, ImageDraw, ImageFont
    from puzzler.task import scene, Variant, beads
    ap = argparse.ArgumentParser()
    ap.add_argument('episode'); ap.add_argument('out'); ap.add_argument('piece')
    ap.add_argument('start', type=int); ap.add_argument('end', type=int); ap.add_argument('camera')
    ap.add_argument('--stride', type=int, default=2); ap.add_argument('--label', default='')
    args = ap.parse_args()
    episode = Path(args.episode)
    result = json.loads((episode / 'result.json').read_text())
    variant = dict(result['variant']); variant['preset_pieces'] = tuple(variant.get('preset_pieces', ()))
    sim = scene(Variant(**variant)).build()
    frames = [f for f in json.loads((episode / 'frames.json').read_text()) if args.start <= f['step'] <= args.end]
    picked = frames[::args.stride]
    poses = [np.asarray(f['poses'][args.piece], float) for f in picked]
    if args.camera == 'auto':
        # Frame the piece's whole path from the front right, slightly above.
        path = np.array([beads(p, args.piece).mean(axis=0) for p in poses])
        lo, hi = path.min(axis=0), path.max(axis=0)
        look = (lo + hi) / 2
        extent = max(float(np.linalg.norm(hi - lo)), .18)
        view = np.array([.78, -.42, .62]); view /= np.linalg.norm(view)
        eye = look + view * (.62 * extent / np.tan(np.radians(FOV) / 2) + .12)
        v = [*eye, *look]
    else:
        v = [float(x) for x in args.camera.split(',')]
    cam = rss.CameraSpec(name='trail', width=W, height=H, fov_y=FOV, eye=tuple(v[:3]), look_at=tuple(v[3:]), up=(0, 0, 1))
    inv = np.linalg.inv(look_matrix(v[:3], v[3:]))
    rots = [Rotation.from_quat(p[3:]) for p in poses]
    turned = np.cumsum([0.] + [np.degrees((rots[i].inv() * rots[i + 1]).magnitude()) for i in range(len(rots) - 1)])
    bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    hz = result['control_hz']
    path_uv, ghosts = [], []
    with iio.get_writer(args.out, fps=30, codec='libx264', quality=8, output_params=['-pix_fmt', 'yuv420p']) as writer:
        for i, f in enumerate(picked):
            sim.state.body_q.assign(np.asarray(f['body_q'], np.float32))
            sim.state.joint_q.assign(np.asarray(f['joint_q'], np.float32))
            base = Image.fromarray(sim.render(cam)['rgb'][..., :3]).convert('RGBA')
            over = Image.new('RGBA', base.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(over)
            pose, rot = poses[i], rots[i].as_matrix()
            center = beads(pose, args.piece).mean(axis=0)
            path_uv.append(tuple(project(center, inv)[0]))
            # leave a faded dial only after a real turn or move, so a resting piece does not pile them up
            if not ghosts or np.degrees(Rotation.from_matrix(ghosts[-1][1].T @ rot).magnitude()) > 8 \
                    or np.linalg.norm(center - ghosts[-1][0]) > .015:
                ghosts.append((center, rot))
            # path: older parts fainter
            for j in range(1, len(path_uv)):
                a = int(70 + 185 * j / len(path_uv))
                d.line([path_uv[j - 1], path_uv[j]], fill=(255, 196, 0, a), width=5)
            # faded dials left behind
            for k, (c, r) in enumerate(ghosts[:-1]):
                a = int(40 + 110 * (k + 1) / max(1, len(ghosts)))
                ring, needle, pin = dial(c, r)
                d.line([tuple(p) for p in project(ring, inv)], fill=(255, 255, 255, a), width=2)
                d.line([tuple(project(c, inv)[0]), tuple(project(needle, inv)[0])], fill=(255, 196, 0, a), width=3)
            # current dial, bold
            ring, needle, pin = dial(center, rot)
            cu = tuple(project(center, inv)[0])
            d.line([tuple(p) for p in project(ring, inv)], fill=(255, 255, 255, 255), width=4)
            d.line([cu, tuple(project(needle, inv)[0])], fill=(255, 196, 0, 255), width=5)
            d.line([cu, tuple(project(pin, inv)[0])], fill=(80, 200, 255, 255), width=5)
            # ring every bead of the piece
            for b in project(beads(pose, args.piece), inv):
                d.ellipse((b[0] - 15, b[1] - 15, b[0] + 15, b[1] + 15), outline=(255, 255, 255, 230), width=3)
            text = f'turned {turned[i]:.0f}° so far'
            d.rounded_rectangle((18, 18, 34 + d.textlength(text, font=bold), 58), 8, fill=(21, 23, 26, 210))
            d.text((26, 22), text, fill=(255, 255, 255, 255), font=bold)
            if args.label:
                d.rounded_rectangle((18, H - 50, 34 + d.textlength(args.label, font=small), H - 18), 8, fill=(21, 23, 26, 200))
                d.text((26, H - 46), args.label, fill=(230, 232, 235, 255), font=small)
            writer.append_data(np.asarray(Image.alpha_composite(base, over).convert('RGB')))
    net = np.degrees((rots[0].inv() * rots[-1]).magnitude())
    print(json.dumps(dict(out=args.out, frames=len(picked), turned_total_deg=round(float(turned[-1]), 1),
                          net_deg=round(float(net), 1), seconds=round((picked[-1]['step'] - picked[0]['step']) / hz, 1))))


if __name__ == '__main__':
    main()
