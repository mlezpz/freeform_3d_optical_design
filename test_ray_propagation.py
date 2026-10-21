import numpy as np

import ray_propagation as rp

def test_set_input_raybundle_keeps_angle_and_sets_speed_to_index():
    coords = rp.Coordinates(xyz_i=(-10.0, -10.0, 0.0),
                            xyz_f=(+10.0, +10.0, 20.0),
                            n_xyz=(5, 5, 5))
    ro = rp.Refractive3dOptic(coords, try_cuda=False)
    glass = rp.FixedIndexMaterial(1.5)
    ro.set_materials(input_material=glass,
                     mixture_materials_list=(glass, glass),
                     output_material=glass)
    theta, phi = np.deg2rad(15), np.deg2rad(45)
    direction = np.array([[np.sin(theta)*np.cos(phi)],
                          [np.sin(theta)*np.sin(phi)],
                          [np.cos(theta)]])
    ro.set_input_raybundle(rp.RayBundle(np.zeros((3, 1)), direction.copy()))

    v_xyz = ro.input_raybundle.v_xyz
    assert np.allclose(np.linalg.norm(v_xyz, axis=0), 1.5)
    assert np.allclose(v_xyz / 1.5, direction)
