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

def make_air_polymer_optic():
    coords = rp.Coordinates(xyz_i=(-10.0, -10.0, 0.0),
                            xyz_f=(+10.0, +10.0, 20.0),
                            n_xyz=(21, 21, 21))
    ro = rp.Refractive3dOptic(coords, try_cuda=False)
    air, polymer = rp.FixedIndexMaterial(1), rp.FixedIndexMaterial(1.5)
    ro.set_materials(input_material=air,
                     mixture_materials_list=(air, polymer),
                     output_material=air)
    return ro

def test_beam_steering_target_tilts_collimated_rays_in_place():
    coords = make_air_polymer_optic().coordinates
    source = rp.TrainingData_for_beam_steering(
        coords, radius=3, max_z_angle_deg=0, x0=0, y0=0, num_rays=20,
        steering_angle_deg=10, azimuth_deg=90)

    input_rays, desired_rays = source.input_output_pair(0.5)

    assert np.allclose(desired_rays.xyz[:2], input_rays.xyz[:2])
    assert np.allclose(desired_rays.z, coords.z_f)
    tilted = np.array([[0], [np.sin(np.deg2rad(10))], [np.cos(np.deg2rad(10))]])
    assert np.allclose(desired_rays.v_xyz, tilted)

def test_beam_steering_target_rotates_tilted_rays_after_straight_travel():
    coords = make_air_polymer_optic().coordinates
    source = rp.TrainingData_for_beam_steering(
        coords, radius=3, max_z_angle_deg=15, x0=0, y0=0, num_rays=20,
        steering_angle_deg=10)

    input_rays, desired_rays = source.input_output_pair(0.5)

    v = input_rays.v_xyz
    travel_time = (coords.z_f - input_rays.z) / v[2]
    assert np.allclose(desired_rays.xyz, input_rays.xyz + travel_time*v)
    a = np.deg2rad(10) # A rotation about +y tilts +z toward +x:
    rotation = np.array([[np.cos(a), 0, np.sin(a)],
                         [0,         1, 0        ],
                         [-np.sin(a), 0, np.cos(a)]])
    assert np.allclose(desired_rays.v_xyz, rotation @ v)

def test_gradient_update_survives_rays_that_hit_their_targets_exactly():
    ro = make_air_polymer_optic()
    # Collimated rays through a uniform optic land exactly where an
    # unsteered target says they should:
    source = rp.TrainingData_for_beam_steering(
        ro.coordinates, radius=3, max_z_angle_deg=0, x0=0, y0=0,
        num_rays=20, steering_angle_deg=0)
    input_rays, desired_rays = source.input_output_pair(0.5)
    ro.set_input_raybundle(input_rays)
    ro.set_desired_output_raybundle(desired_rays)

    ro.gradient_update(step_size=1, z_planes=(0, 1), smoothing_sigma=1, dt=0.2)

    ro.update_attributes()
    assert np.all(np.isfinite(ro.concentration))
