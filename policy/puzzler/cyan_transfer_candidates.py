"""Finite public body-relative Cyan receiving-grasp proposals.

No saved world poses, joint states, clearance claims or observations are loaded.
All candidates require fresh runtime IK and complete collision screening.
"""
import numpy as np
from scipy.spatial.transform import Rotation

# bead, bead-to-TCP axial depth, body approach unit vector, local roll degrees.
FAMILY = [[2, 0.008, [1.0, 0.0, 0.0], 180], [2, 0.014, [1.0, 0.0, 0.0], 180], [2, 0.014, [1.0, 0.0, 0.0], 210], [2, 0.008, [1.0, 0.0, 0.0], 210], [2, 0.014, [0.75, 0.43301270189221924, 0.5000000000000001], 210], [2, 0.014, [0.8660254037844386, 0.0, 0.5000000000000001], 270], [2, 0.014, [0.8660254037844386, 0.0, 0.5000000000000001], 240], [2, 0.014, [0.7499999999999997, -0.4330127018922197, 0.5000000000000001], 330], [2, 0.014, [1.0, 0.0, 0.0], 270], [2, 0.014, [0.8660254037844386, 0.0, 0.5000000000000001], 210], [2, 0.014, [0.8660254037844386, 0.0, 0.5000000000000001], 150], [2, 0.014, [0.8660254037844386, 0.0, 0.5000000000000001], 180], [2, 0.008, [0.75, 0.43301270189221924, 0.5000000000000001], 210], [2, 0.014, [0.75, 0.43301270189221924, 0.5000000000000001], 180], [2, 0.014, [0.7499999999999997, -0.4330127018922197, 0.5000000000000001], 180], [2, 0.014, [0.49999999999999994, 0.0, 0.8660254037844387], 270], [2, 0.014, [0.8660254037844384, -0.5000000000000004, 0.0], 330], [2, 0.014, [0.7499999999999997, -0.4330127018922197, 0.5000000000000001], 150], [2, 0.014, [0.7499999999999997, -0.4330127018922197, 0.5000000000000001], 210], [2, 0.008, [0.8660254037844386, 0.0, 0.5000000000000001], 240], [2, 0.014, [1.0, 0.0, 0.0], 240], [2, 0.008, [0.49999999999999994, 0.0, 0.8660254037844387], 270], [2, 0.014, [0.75, 0.43301270189221924, 0.5000000000000001], 150], [2, 0.0, [1.0, 0.0, 0.0], 180], [2, 0.008, [0.8660254037844384, -0.5000000000000004, 0.0], 330], [2, 0.008, [1.0, 0.0, 0.0], 270], [2, 0.008, [0.8660254037844386, 0.0, 0.5000000000000001], 270], [2, 0.014, [0.43301270189221913, -0.25000000000000017, 0.8660254037844387], 300], [2, 0.0, [0.75, 0.43301270189221924, 0.5000000000000001], 210], [2, 0.0, [1.0, 0.0, 0.0], 210], [2, 0.008, [0.8660254037844386, 0.0, 0.5000000000000001], 210], [2, 0.008, [0.8660254037844386, 0.0, 0.5000000000000001], 150], [2, 0.008, [0.8660254037844386, 0.0, 0.5000000000000001], 180], [2, 0.008, [0.7499999999999997, -0.4330127018922197, 0.5000000000000001], 330]]


def proposal_family(local_beads):
    beads=np.asarray(local_beads,dtype=float)
    if beads.shape!=(3,3) or not np.isfinite(beads).all():
        raise ValueError('Public Cyan CAD must be the finite three-bead geometry')
    output=[]
    for bead,depth,approach,roll in FAMILY:
        z=np.asarray(approach,dtype=float);z/=np.linalg.norm(z)
        axis=np.array([0.,0.,1.]) if abs(z[2])<.9 else np.array([1.,0.,0.])
        x=np.cross(axis,z);x/=np.linalg.norm(x);y=np.cross(z,x)
        H=np.column_stack([x,y,z])@Rotation.from_euler('z',roll,degrees=True).as_matrix()
        p=beads[bead]-H@np.array([0.,0.,depth])
        output.append(dict(bead_index=bead,hand_in_body_position=p.tolist(),
            hand_in_body_rotation=H.tolist(),bead_in_tcp_m=[0.,0.,depth],
            nominal_contact_width_m=.022,source_open_width_m=.055,
            permitted_distal_bead_indices=[bead],proposal_only=True,
            geometry_or_IK_prevalidated_at_runtime=False))
    return output
