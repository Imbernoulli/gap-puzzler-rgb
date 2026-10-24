"""Explicit compound colliders: bead cavities and piece concavities stay open."""
import json
from pathlib import Path
import numpy as np
from .model import ROOT, CELLS, RADIUS, PITCH, local_beads

def generate():
    import trimesh as tm
    folder=ROOT/'assets';folder.mkdir(exist_ok=True)
    manifest={}
    def save(name,meshes):
        visual=folder/f'{name}.obj'
        tm.util.concatenate(meshes).export(visual)
        proxies=[]
        for i,m in enumerate(meshes):
            p=folder/f'{name}_{i:03d}.obj';m.export(p);proxies.append(str(p))
        manifest[name]=dict(path=str(visual),collision_path=proxies[0],collision_paths=proxies[1:])
    for name in CELLS:
        meshes=[]
        centers=local_beads(name)
        for p in centers:
            s=tm.creation.icosphere(subdivisions=2,radius=RADIUS);s.apply_translation(p);meshes.append(s)
        for i,a in enumerate(centers):
            for b in centers[i+1:]:
                delta=b-a
                if abs(np.linalg.norm(delta)-PITCH)>1e-7:continue
                neck=tm.creation.cylinder(radius=.0045,height=PITCH,sections=16)
                neck.apply_transform(tm.geometry.align_vectors([0,0,1],delta))
                neck.apply_translation((a+b)/2);meshes.append(neck)
        save(name,meshes)
    def box(ext,at):
        m=tm.creation.box(extents=ext);m.apply_translation(at);return m
    pieces=[box((.146,.296,.006),(0,0,0))]
    # A shallow socket lattice: bead centers are surrounded by four rounded posts.
    for i in range(6):
        for j in range(12):
            p=((i-2.5)*PITCH,(j-5.5)*PITCH,.007)
            s=tm.creation.cylinder(radius=.006,height=.008,sections=12);s.apply_translation(p);pieces.append(s)
    pieces += [box((.009,.315,.016),(a,0,.005)) for a in (-.078,.078)]
    pieces += [box((.165,.009,.016),(0,a,.005)) for a in (-.153,.153)]
    # The video's hinge and open lid border; no added grasping fixture.
    hinge=tm.creation.cylinder(radius=.013,height=.315,sections=24)
    hinge.apply_transform(tm.transformations.rotation_matrix(np.pi/2,[1,0,0]))
    hinge.apply_translation((.095,0,.014));pieces.append(hinge)
    pieces.append(box((.045,.315,.004),(.115,0,.003)))
    save('board',pieces)
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

def manifest():
    """Resolve archived mesh names within this checkout, without regeneration.

    The copied manifest contains historical absolute paths. Their 191 meshes
    were checked byte-for-byte against this checkout before this relocation;
    see results/reviews/asset_path_relocation_v1.json. Runtime resolution reads
    only the local manifest and never follows those historical paths.
    """
    root=ROOT.resolve()
    folder=root/'assets'
    if folder.is_symlink() or folder.resolve().parent!=root:
        raise ValueError('Assets must be an ordinary directory in the current checkout')
    p=folder/'manifest.json'
    if p.is_symlink():
        raise ValueError('Asset manifest must be an ordinary local file')
    if not p.is_file():
        raise FileNotFoundError(f'Missing local asset manifest: {p}; generate assets explicitly')
    value=json.loads(p.read_text())
    if not isinstance(value,dict) or set(value)!=set(CELLS)|{'board'}:
        raise ValueError('Asset manifest must contain the twelve known pieces and board')
    def local(reference,expected):
        if not isinstance(reference,str) or not reference or '\\' in reference:
            raise ValueError(f'Invalid mesh reference for {expected}')
        declared=Path(reference)
        if ('..' in declared.parts or declared.name!=expected or
                (declared.is_absolute() and declared.parent.name!='assets') or
                (not declared.is_absolute() and declared.parts not in ((expected,),('assets',expected)))):
            raise ValueError(f'Unexpected mesh reference for {expected}: {reference}')
        path=folder/expected
        if path.is_symlink() or path.resolve().parent!=folder:
            raise ValueError(f'Local mesh must be an ordinary file in assets: {path}')
        if not path.is_file():
            raise FileNotFoundError(f'Missing local mesh: {path}; refusing geometry regeneration')
        return str(path)
    normalized={}
    for name,record in value.items():
        if not isinstance(record,dict) or not isinstance(record.get('collision_paths'),list):
            raise ValueError(f'Invalid collider manifest for {name}')
        normalized[name]={**record,
            'path':local(record.get('path'),f'{name}.obj'),
            'collision_path':local(record.get('collision_path'),f'{name}_000.obj'),
            'collision_paths':[local(reference,f'{name}_{index:03d}.obj')
                               for index,reference in enumerate(record['collision_paths'],1)]}
    return normalized
