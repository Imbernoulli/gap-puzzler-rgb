"""Explicit simulated finger extensions; stock YAM collisions/gains are retained.

This is a custom hardware assumption, not the stock gripper or video hardware.
Each added finger is a 5 g rigid stem with two narrow contact rails. No object
attachment, contact disabling, force multiplier or actuator retuning is used.
"""
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .model import ROOT

NAME = 'yam_bimanual_precision40v2'
HAND = 'yam_precision40v2'


def register(name=NAME):
    import robosimstudio
    from robosimstudio.specs import AssetSpec
    from robosimstudio.robots.registry import (
        EMBODIMENTS, register_embodiment, register_gripper,
    )
    from robosimstudio.robots.embodiments.yam_bimanual import YAM_BIMANUAL
    from robosimstudio.robots.grippers.yam_linear_arena import YAM_LINEAR_ARENA

    lengths={'yam_bimanual_precision40v2':.040,'yam_bimanual_precision20v1':.020}
    if name not in lengths:raise ValueError(f'Unknown custom embodiment: {name}')
    extension=lengths[name]
    hand_name=name.replace('yam_bimanual_','yam_')
    if name in EMBODIMENTS.names():
        return name
    original = Path(robosimstudio.__file__).resolve().parent.parent / 'assets/embodiments/yam/yam_bimanual_arms.xml'
    tree = ET.parse(original)
    root = tree.getroot()
    root.set('model', name)
    root.find('compiler').set('meshdir', str(original.parent / 'i2rt_yam/assets'))
    for arm in ['left', 'right']:
        for side, sign in [('left', 1.), ('right', -1.)]:
            finger = root.find(f'.//body[@name="{arm}_link_{side}_finger"]')
            origin = np.fromstring(finger.get('pos'), sep=' ')
            quat = np.fromstring(finger.get('quat'), sep=' ')
            rotation = Rotation.from_quat(np.r_[quat[1:], quat[0]]).as_matrix()
            zshift=extension-.040
            center = np.array([sign * .003, -.044, .151+zshift/2])
            local_pos = rotation.T @ (center - origin)
            local_q = Rotation.from_matrix(rotation.T).as_quat()
            child = ET.SubElement(finger, 'body', {
                'name': f'{arm}_{side}_precision_extension',
                'pos': ' '.join(map(str, local_pos)),
                'quat': ' '.join(map(str, np.r_[local_q[3], local_q[:3]])),
                'gravcomp': '1',
            })
            # Conservative rectangular-envelope inertia for a 5 g, 7x14x45 mm tip.
            ET.SubElement(child, 'inertial', {
                'pos': '0 0 .004', 'mass': '.005',
                'diaginertia': '9.254e-7 8.642e-7 1.021e-7',
            })
            def capsule(label, a, b, radius):
                points = np.r_[np.asarray(a)-center, np.asarray(b)-center]
                ET.SubElement(child, 'geom', {
                    'name': f'{arm}_{side}_{label}', 'type': 'capsule',
                    'fromto': ' '.join(map(str, points)), 'size': str(radius),
                    'friction': '2.0 0.1 0.01', 'condim': '6',
                    'solimp': '.998 .998 .001', 'solref': '.004 1',
                    'rgba': '.55 .65 .75 1', 'group': '3',
                })
            capsule('extension_stem', [sign*.003, -.044, .132], [sign*.003, -.044, .165+zshift], .002)
            for i, y in enumerate([-.049, -.039]):
                capsule(f'precision_rail{i}', [sign*.0015, y, .160+zshift], [sign*.0015, y, .177+zshift], .0015)
    asset = ROOT / f'assets/{hand_name}.xml'
    tree.write(asset, encoding='unicode')
    # Fixed children collapse into the existing finger bodies. Their collision
    # shapes therefore use the original two-sided contact body mapping.
    register_gripper(replace(YAM_LINEAR_ARENA, name=hand_name, tcp_offset=(0., -.044, .130+extension),
                             tip_reach=.0085))
    register_embodiment(replace(YAM_BIMANUAL, name=name, asset=AssetSpec(path=str(asset)),
                               arms=tuple(replace(a, gripper=hand_name) for a in YAM_BIMANUAL.arms)))
    return name
