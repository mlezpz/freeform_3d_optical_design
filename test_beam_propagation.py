import numpy as np
import pytest

import beam_propagation as bp

@pytest.mark.parametrize('theta_deg', [0, 15])
def test_propagation_through_a_uniform_region_matches_angular_spectrum(
    theta_deg):
    coords = bp.Coordinates(xyz_i=(-12.7, -12.7, 0.0),
                            xyz_f=(+12.7, +12.7, 10.0),
                            n_xyz=(64, 64, 16))
    ro = bp.Refractive3dOptic(coords, try_cuda=False)
    ro.set_materials((bp.FixedIndexMaterial(1), bp.FixedIndexMaterial(1.5)))
    # The beam sees a uniform index of 1.27, which falls between
    # reference indices. The corner voxels, far from the beam, force
    # the reference indices to span the full 1.0-1.5 range:
    concentration = np.full((16, 64, 64), 0.54)
    concentration[:, 0, 0], concentration[:, -1, -1] = 0, 1
    ro.set_3d_concentration(concentration)
    wavelength = 1.0
    input_field, _ = bp.TrainingData_for_2dImaging(
        coords, radius=1).input_output_pair(
            0, 0, wavelength, divergence_angle_degrees=10,
            theta=np.deg2rad(theta_deg))
    ro.set_2d_input_field(input_field, wavelength)

    ro._calculate_3d_field()

    k = 2*np.pi/wavelength
    kx = 2*np.pi*np.fft.fftfreq(64, coords.dx).reshape(1, 64)
    ky = 2*np.pi*np.fft.fftfreq(64, coords.dy).reshape(64, 1)
    kz = np.sqrt((1.27*k)**2 - kx**2 - ky**2 + 0j)
    expected = np.fft.ifft2(
        np.exp(1j*kz*16*coords.dz) * np.fft.fft2(input_field))
    calculated = ro._to_numpy(ro._calculated_field_tensor[-1])
    error = np.linalg.norm(calculated - expected) / np.linalg.norm(expected)
    assert error < 5e-3

def make_trainable_optic():
    coords = bp.Coordinates(xyz_i=(-12.7, -12.7, 0.0),
                            xyz_f=(+12.7, +12.7, 5.0),
                            n_xyz=(32, 32, 5))
    ro = bp.Refractive3dOptic(coords, try_cuda=False)
    ro.set_materials((bp.FixedIndexMaterial(1), bp.FixedIndexMaterial(1.5)))
    ro.set_3d_concentration()
    input_field, desired_output_field = bp.TrainingData_for_2dImaging(
        coords, radius=3).input_output_pair(1, 0, 1.0, 15)
    ro.set_2d_input_field(input_field, 1.0)
    ro.set_2d_desired_output_field(desired_output_field)
    return ro

def test_gradient_update_with_momentum_applies_nesterov_steps():
    ro = make_trainable_optic()
    step, mu, sigma = 10, 0.9, 2
    ro.gradient_update(step, smoothing_sigma=sigma, momentum=mu)
    g1 = [bp.smooth_2d(g, sigma=sigma) for g in ro._gradient_tensor]
    before = [c.clone() for c in ro._composition_tensor]

    ro.gradient_update(step, smoothing_sigma=sigma, momentum=mu)

    g2 = [bp.smooth_2d(g, sigma=sigma) for g in ro._gradient_tensor]
    for c_before, c_after, g1_c, g2_c in zip(
        before, ro._composition_tensor, g1, g2):
        velocity = mu*g1_c + g2_c
        expected = c_before - step*(g2_c + mu*velocity)
        assert np.allclose(c_after.numpy(), expected.numpy())

def test_set_3d_concentration_resets_momentum():
    ro = make_trainable_optic()
    ro.gradient_update(10, momentum=0.9)
    assert hasattr(ro, '_velocity_tensor')

    ro.set_3d_concentration()

    assert not hasattr(ro, '_velocity_tensor')
