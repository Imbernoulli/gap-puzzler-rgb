"""Draw a rig's policy cameras as frustums in a rendered view of the workcell.

  camera_rig_figure.py EPISODE_DIR OUT.png official|custom [OBS_INDEX]

Run inside the run's workspace (RSS_WORKSPACE). Camera poses are the policy's
own: for the official rig the per-frame ``camera_to_world`` the observer saved
(gantry fixed, wrists from encoder FK); for the custom rig the fixed rig spec.
"""
import json
import sys
from pathlib import Path

import numpy as np

VIEWER = dict(eye=(2.05, -1.75, 2.35), look_at=(.40, .0, 1.02), fov_y=38, width=1600, height=1000)
COLORS = {'top': (230, 90, 40), 'front': (40, 120, 220), 'rear': (40, 170, 110), 'side': (150, 80, 200),
          'left_wrist': (40, 120, 220), 'right_wrist': (40, 170, 110)}
LABELS = {'top': 'top', 'front': 'front', 'rear': 'rear', 'side': 'side',
          'left_wrist': 'left wrist', 'right_wrist': 'right wrist'}


def look_matrix(eye, look_at, up=(0., 0., 1.)):
    """Camera-to-world in the vision basis: +z forward, +x right, +y down."""
    eye, look_at, up = map(lambda v: np.asarray(v, float), (eye, look_at, up))
    f = look_at - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, d, f, eye
    return m


def project(points, m, fov_y, width, height):
    inv = np.linalg.inv(m)
    p = (inv[:3, :3] @ np.asarray(points, float).T + inv[:3, 3:4]).T
    fpx = .5 * height / np.tan(np.radians(fov_y) / 2)
    return np.c_[width / 2 + fpx * p[:, 0] / p[:, 2], height / 2 + fpx * p[:, 1] / p[:, 2]], p[:, 2]


def frustum(m, fov_y, aspect, depth):
    ty = np.tan(np.radians(fov_y) / 2); tx = ty * aspect
    c, r, d, f = m[:3, 3], m[:3, 0], m[:3, 1], m[:3, 2]
    corners = [c + depth * (f + sx * tx * r + sy * ty * d) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return c, corners


def main():
    import robosimstudio as rss
    from PIL import Image, ImageDraw, ImageFont
    from puzzler.task import scene, Variant
    from puzzler import vision
    episode, out, rig = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    index = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    result = json.loads((episode / 'result.json').read_text())
    variant = dict(result['variant']); variant['preset_pieces'] = tuple(variant.get('preset_pieces', ()))
    sim = scene(Variant(**variant)).build()
    observations = json.loads((episode / 'rgb/observations.json').read_text())
    frames = json.loads((episode / 'frames.json').read_text())
    stamp = observations[index]['timestamp']
    frame = min(frames, key=lambda f: abs(f['step'] - stamp * result['control_hz']))
    sim.state.body_q.assign(np.asarray(frame['body_q'], np.float32))
    sim.state.joint_q.assign(np.asarray(frame['joint_q'], np.float32))
    view = rss.CameraSpec(name='viewer', up=(0, 0, 1), **VIEWER)
    image = Image.fromarray(sim.render(view)['rgb'][..., :3]).convert('RGBA')
    viewer = look_matrix(VIEWER['eye'], VIEWER['look_at'])
    cams = []
    if rig == 'official':
        poses = observations[index]['robot_geometry_provenance']['camera_to_world']
        for name, m in poses.items():
            cams.append((name, np.asarray(m, float), vision.D405['fov_y'], 1280 / 720, .13 if 'wrist' in name else .30))
    else:
        for spec in vision.rig_specs():
            m = look_matrix(spec.eye, spec.look_at, spec.up)
            cams.append((spec.name, m, spec.fov_y, spec.width / spec.height, .24))
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
    W, H = VIEWER['width'], VIEWER['height']
    for name, m, fov, aspect, depth in cams:
        apex, corners = frustum(m, fov, aspect, depth)
        uv, z = project([apex, *corners], viewer, VIEWER['fov_y'], W, H)
        col = COLORS.get(name, (220, 220, 220))
        pts = [tuple(p) for p in uv]
        draw.polygon(pts[1:], fill=col + (45,), outline=col + (255,))
        for p in pts[1:]:
            draw.line([pts[0], p], fill=col + (255,), width=4)
        draw.polygon(pts[1:], outline=col + (255,), width=4)
        draw.ellipse((pts[0][0] - 9, pts[0][1] - 9, pts[0][0] + 9, pts[0][1] + 9), fill=col + (255,), outline=(255, 255, 255, 255), width=2)
        label = LABELS.get(name, name)
        tw = draw.textlength(label, font=font)
        lx, ly = pts[0][0] + 14, pts[0][1] - 34
        if name == 'left_wrist':  # keep the two wrist labels apart
            lx, ly = pts[0][0] - tw - 20, pts[0][1] - 40
        draw.rounded_rectangle((lx - 6, ly - 3, lx + tw + 6, ly + 30), 6, fill=(21, 23, 26, 220))
        draw.text((lx, ly), label, fill=(255, 255, 255, 255), font=font)
        print(name, 'apex', np.round(apex, 3), 'forward', np.round(m[:3, 2], 3), 'in view', bool(np.all(z > 0)))
    final = Image.alpha_composite(image, overlay).convert('RGB')
    # Crop the empty backdrop: keep everything that differs from the corner colour, plus a margin.
    a = np.asarray(final).astype(int)
    mask = np.abs(a - a[2, 2]).sum(axis=2) > 24
    ys, xs = np.nonzero(mask)
    x0, x1 = max(0, xs.min() - 30), min(W, xs.max() + 30)
    y0, y1 = max(0, ys.min() - 30), min(H, ys.max() + 30)
    final.crop((x0, y0, x1, y1)).save(out, quality=88)
    print(out, 'frame step', frame['step'], 't', stamp)


if __name__ == '__main__':
    main()
