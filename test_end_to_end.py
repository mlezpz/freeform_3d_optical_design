import json

import numpy as np
import pytest
import tifffile as tf

import beam_propagation as bp
import ray_propagation as rp
from end_to_end import design_and_export, export_index_volume

def make_ray_optic(n=21):
    coords = rp.Coordinates(xyz_i=(-10.0, -10.0, 0.0),
                            xyz_f=(+10.0, +10.0, 20.0),
                            n_xyz=(n, n, n))
    ro = rp.Refractive3dOptic(coords, try_cuda=False)
    air, polymer = rp.FixedIndexMaterial(1), rp.FixedIndexMaterial(1.5)
    ro.set_materials(input_material=air,
                     mixture_materials_list=(air, polymer),
                     output_material=air)
    return ro

def make_beam_optic(nz=5):
    coords = bp.Coordinates(xyz_i=(-12.7, -12.7, 0.0),
                            xyz_f=(+12.7, +12.7, 5.0),
                            n_xyz=(32, 32, nz))
    ro = bp.Refractive3dOptic(coords, try_cuda=False)
    ro.set_materials((bp.FixedIndexMaterial(1), bp.FixedIndexMaterial(1.5)))
    return ro

def z_ramp(coords):
    nx, ny, nz = map(int, coords.n_xyz)
    ramp = np.linspace(0.1, 0.9, nz).reshape(nz, 1, 1)
    return np.broadcast_to(ramp, (nz, ny, nx)).copy()

@pytest.mark.parametrize('make_optic', [make_ray_optic, make_beam_optic])
def test_export_writes_index_concentration_and_metadata(make_optic, tmp_path):
    ro = make_optic()
    concentration = z_ramp(ro.coordinates)
    ro.set_3d_concentration(concentration)

    export_index_volume(ro, 0.5, tmp_path, extra_metadata={'note': 'hi'})

    index = tf.imread(tmp_path / 'index.tif')
    np.testing.assert_allclose(index, 1 + 0.5*concentration, rtol=1e-6)
    np.testing.assert_allclose(tf.imread(tmp_path / 'concentration.tif'),
                               concentration, rtol=1e-6)
    metadata = json.loads((tmp_path / 'metadata.json').read_text())
    c = ro.coordinates
    assert metadata['engine'] == type(ro).__module__
    assert metadata['shape_zyx'] == list(concentration.shape)
    assert metadata['first_voxel_center_xyz'] == [float(v) for v in c.xyz_i]
    assert metadata['voxel_size_xyz'] == [float(v) for v in c.d_xyz]
    assert metadata['index_min'] == pytest.approx(1.05)
    assert metadata['index_max'] == pytest.approx(1.45)
    assert metadata['mixture_materials'] == [
        {'type': 'FixedIndexMaterial', 'index': 1.0},
        {'type': 'FixedIndexMaterial', 'index': 1.5}]
    assert metadata['note'] == 'hi'

def test_export_records_ray_engine_input_output_materials(tmp_path):
    export_index_volume(make_ray_optic(), 0.5, tmp_path)
    metadata = json.loads((tmp_path / 'metadata.json').read_text())
    assert metadata['input_material'] == {
        'type': 'FixedIndexMaterial', 'index': 1.0}
    assert metadata['output_material'] == {
        'type': 'FixedIndexMaterial', 'index': 1.0}

def test_design_and_export_with_ray_engine(tmp_path):
    ro = make_ray_optic()
    data_source = rp.TrainingData_for_2dImaging(
        ro.coordinates, radius=3, max_z_angle_deg=15, x0=0, y0=0, num_rays=100)

    def set_training_pair(optic):
        input_rays, desired_output_rays = data_source.input_output_pair(0.5)
        optic.set_input_raybundle(input_rays)
        optic.set_desired_output_raybundle(desired_output_rays)

    loss_history = design_and_export(
        ro, set_training_pair, 0.5, iterations=3, output_dir=tmp_path,
        dt=0.2, step_size=10, z_planes=(0, 1), smoothing_sigma=1)

    check_design_outputs(loss_history, tmp_path)

def test_design_and_export_with_beam_engine(tmp_path):
    ro = make_beam_optic()
    data_source = bp.TrainingData_for_2dImaging(ro.coordinates, radius=3)

    def set_training_pair(optic):
        input_field, desired_output_field = data_source.input_output_pair(
            0, 0, wavelength=1, divergence_angle_degrees=15)
        optic.set_2d_input_field(input_field, 1)
        optic.set_2d_desired_output_field(desired_output_field)

    loss_history = design_and_export(
        ro, set_training_pair, 1, iterations=3, output_dir=tmp_path,
        step_size=100, z_planes=(1,), smoothing_sigma=1)

    check_design_outputs(loss_history, tmp_path)

def check_design_outputs(loss_history, output_dir):
    assert len(loss_history) == 3
    assert np.all(np.isfinite(loss_history))
    concentration = tf.imread(output_dir / 'concentration.tif')
    assert not np.allclose(concentration, 0.5)
    metadata = json.loads((output_dir / 'metadata.json').read_text())
    assert metadata['iterations'] == 3
    assert metadata['loss_history'] == loss_history
