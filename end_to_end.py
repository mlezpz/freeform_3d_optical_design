import json
from pathlib import Path

import numpy as np

"""
Design a freeform 3D refractive optic and export a self-describing,
fabrication-ready refractive index volume, in one call.

Works with the `Refractive3dOptic` from either `ray_propagation` or
`beam_propagation`. Run this module for a small ray-optics demo.
"""

def design_and_export(optic,
                      set_training_pair,
                      wavelength_um,
                      iterations,
                      output_dir,
                      **gradient_update_kwargs):
    """Optimize `optic`, then export it via `export_index_volume()`.

    `set_training_pair(optic)` is called once per iteration, and should
    set the optic's input and desired output (rays or fields, depending
    on the engine). `gradient_update_kwargs` are passed straight to
    `optic.gradient_update()`.
    """
    loss_history = []
    for _ in range(iterations):
        set_training_pair(optic)
        optic.gradient_update(**gradient_update_kwargs)
        loss_history.append(float(optic.loss))
    export_index_volume(optic, wavelength_um, output_dir, extra_metadata={
        'iterations': iterations,
        'gradient_update_kwargs': gradient_update_kwargs,
        'loss_history': loss_history})
    return loss_history

def export_index_volume(optic, wavelength_um, output_dir, extra_metadata=None):
    """Write `index.tif`, `concentration.tif` and `metadata.json`.

    The TIFs are float32 ZYX stacks with voxel size embedded for
    ImageJ. Lengths are in the units of `optic.coordinates` (microns, if
    you're using a SellmeierMaterial).
    """
    import tifffile as tf
    optic.update_attributes()
    concentration = np.asarray(optic.concentration, dtype='float64')
    n0, n1 = [m.get_index(wavelength_um) for m in optic.material_list]
    index = n0 + (n1 - n0)*concentration

    c = optic.coordinates
    dx, dy, dz = map(float, c.d_xyz)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, volume in (('index', index), ('concentration', concentration)):
        tf.imwrite(output_dir / (name + '.tif'), volume.astype('float32'),
                   imagej=True, resolution=(1/dx, 1/dy),
                   metadata={'spacing': dz, 'axes': 'ZYX'})

    metadata = {
        'engine': type(optic).__module__,
        'axis_order': 'zyx',
        'shape_zyx': list(index.shape),
        'first_voxel_center_xyz': [float(v) for v in c.xyz_i],
        'voxel_size_xyz': [dx, dy, dz],
        'wavelength_um': float(wavelength_um),
        'index_min': float(index.min()),
        'index_max': float(index.max()),
        'mixture_materials': [_describe(m) for m in optic.material_list],
        }
    for name in ('input_material', 'output_material'):
        if hasattr(optic, name):
            metadata[name] = _describe(getattr(optic, name))
    metadata.update(extra_metadata or {})
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    return index

def _describe(material):
    return {'type': type(material).__name__, **vars(material)}

if __name__ == '__main__':
    from ray_propagation import (
        Coordinates, Refractive3dOptic, FixedIndexMaterial,
        TrainingData_for_2dImaging)

    coords = Coordinates(xyz_i=(-10.0, -10.0,   0.0),
                         xyz_f=(+10.0, +10.0, +20.0),
                         n_xyz=(   41,    41,    41))
    ro = Refractive3dOptic(coords)
    air, polymer = FixedIndexMaterial(1), FixedIndexMaterial(1.5)
    ro.set_materials(input_material=air,
                     mixture_materials_list=(air, polymer),
                     output_material=air)
    wavelength_um = 0.5
    data_source = TrainingData_for_2dImaging(
        coords, radius=3, max_z_angle_deg=15, x0=0, y0=0, num_rays=1000)

    def set_training_pair(optic):
        input_rays, desired_output_rays = (
            data_source.input_output_pair(wavelength_um))
        optic.set_input_raybundle(input_rays)
        optic.set_desired_output_raybundle(desired_output_rays)

    output_dir = Path(__file__).parent / 'output' / 'end_to_end'
    loss_history = design_and_export(
        ro, set_training_pair, wavelength_um, iterations=30,
        output_dir=output_dir,
        dt=0.1, step_size=10, z_planes=(0, 1), smoothing_sigma=2)
    print("Loss went from %0.4f to %0.4f"%(loss_history[0], loss_history[-1]))
    print("Wrote index.tif, concentration.tif, metadata.json to", output_dir)
