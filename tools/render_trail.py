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


def arrow(d, a, b, color, width):
    """A 2D arrow from a to b with a filled head."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    v = b - a; n = np.linalg.norm(v)
    if n < 1:
        return
    u = v / n; head = min(14 + width, .45 * n)
    base = b - u * head; side = np.array([-u[1], u[0]]) * head * .45
    d.line([tuple(a), tuple(base)], fill=color, width=width)
    d.polygon([tuple(b), tuple(base + side), tuple(base - side)], fill=color)


def triad(d, center, rot, inv, length, width, alpha):
    c = project(center, inv)[0]
    for k, col in enumerate(((235, 60, 50), (60, 190, 80), (60, 120, 240))):
        arrow(d, c, project(center + length * rot[:, k], inv)[0], col + (alpha,), width)


def piece_copy(d, pts, inv, color, alpha, outline):
    """Draw the piece's beads at their projected size, necks between neighbours."""
    cam = (inv[:3, :3] @ pts.T + inv[:3, 3:4]).T
    uv = project(pts, inv)
    fpx = .5 * H / np.tan(np.radians(FOV) / 2)
    r = fpx * .0125 / cam[:, 2]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if abs(np.linalg.norm(pts[i] - pts[j]) - .025) < .004:
                d.line([tuple(uv[i]), tuple(uv[j])], fill=color + (alpha,), width=max(2, int(.7 * min(r[i], r[j]))))
    for k in np.argsort(-cam[:, 2]):
        x, y, rr = uv[k][0], uv[k][1], r[k]
        d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=color + (alpha,), outline=outline, width=2)


def main():
    import imageio.v2 as iio
    import robosimstudio as rss
    from PIL import Image, ImageDraw, ImageFont
    from puzzler.task import scene, Variant, beads
    from puzzler.model import COLORS
    ap = argparse.ArgumentParser()
    ap.add_argument('episode'); ap.add_argument('out'); ap.add_argument('piece')
    ap.add_argument('start', type=int); ap.add_argument('end', type=int); ap.add_argument('camera')
    ap.add_argument('--stride', type=int, default=2); ap.add_argument('--label', default='')
    args = ap.parse_args()
    episode = Path(args.episode)
    result = json.loads((episode / 'result.json').read_text())
    variant = dict(result['variant']); variant['preset_pieces'] = tuple(variant.get('preset_pieces', ()))
    if variant.get('appearance') == 'withheld':  # the policy is not told the colours; the scene's are in the manifest
        variant['appearance'] = json.loads((episode.parent / 'manifest.json').read_text())['scene_variant']['appearance']
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
        eye = look + view * (.47 * extent / np.tan(np.radians(FOV) / 2) + .12)
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
    color = tuple(int(255 * c) for c in COLORS[args.piece])
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
            # leave a faded copy only after a real turn or move, so a resting piece does not pile them up
            if not ghosts or np.degrees(Rotation.from_matrix(ghosts[-1][1].T @ rot).magnitude()) > 15 \
                    or np.linalg.norm(center - ghosts[-1][0]) > .030:
                ghosts.append((center, rot, beads(pose, args.piece)))
            # path: older parts fainter
            for j in range(1, len(path_uv)):
                a = int(60 + 150 * j / len(path_uv))
                d.line([path_uv[j - 1], path_uv[j]], fill=(255, 210, 60, a), width=3)
            # earlier poses: the piece's triad left behind at each one
            n = max(1, len(ghosts) - 1)
            for k, (c, r, pts) in enumerate(ghosts[:-1]):
                a = int(55 + 110 * (k + 1) / n)
                triad(d, c, r, inv, .045, 3, min(255, a + 60))
            # current pose: outline its beads and draw a larger triad
            pts = beads(pose, args.piece)
            cam_pts = project(pts, inv)
            for b_ in cam_pts:
                d.ellipse((b_[0] - 4, b_[1] - 4, b_[0] + 4, b_[1] + 4), fill=(255, 255, 255, 230))
            triad(d, center, rot, inv, .075, 6, 255)
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
