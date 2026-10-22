import json
import time
from pathlib import Path

import numpy as np

from end_to_end import design_and_export, half_pitch_grin_concentration

"""
A Streamlit UI for designing a 3D refractive optic for one of the built-in
targets (an inverted image, or beam steering for the ray engine), and for
viewing the exported results.

    streamlit run app.py
"""

OUTPUT_ROOT = Path(__file__).parent / 'output'

# Each engine's defaults come from its own example code, on a smaller grid:
DEFAULTS = {
    'ray_propagation': dict(
        half_width=10.0, z_length=20.0, n_xy=41, n_z=41, wavelength_um=0.5,
        step_size=10.0, z_planes=(0, 1), smoothing_sigma=2.0, radius=3.0,
        max_z_angle_deg=15.0, num_rays=1000, dt=0.1),
    'beam_propagation': dict(
        half_width=12.7, z_length=25.4, n_xy=64, n_z=32, wavelength_um=1.0,
        step_size=10.0, momentum=0.9, z_planes=(1, 2, 3), smoothing_sigma=5.0,
        radius=3.0, divergence_angle_degrees=15.0),
    }

# What each target changes from its engine's defaults. Steering is judged
# in the far field, needs a smaller step, and has no GRIN shortcut:
TARGETS = {
    'Inverted image': {},
    'Beam steering': dict(
        z_planes=(20, 40), step_size=0.1, max_z_angle_deg=0.0, start='Uniform',
        steering_angle_deg=10.0, azimuth_deg=0.0),
    }

def build_run(params):
    """Turn form values into (optic, set_training_pair, gradient_update_kwargs)
    """
    if params['engine'] == 'ray_propagation':
        import ray_propagation as engine
    else:
        import beam_propagation as engine
    h, n_xy = params['half_width'], params['n_xy']
    coords = engine.Coordinates(xyz_i=(-h, -h, 0.0),
                                xyz_f=(+h, +h, params['z_length']),
                                n_xyz=(n_xy, n_xy, params['n_z']))
    optic = engine.Refractive3dOptic(coords)
    base = engine.FixedIndexMaterial(params['base_index'])
    mixer = engine.FixedIndexMaterial(params['mixer_index'])
    wavelength = params['wavelength_um']
    gradient_update_kwargs = dict(
        step_size=params['step_size'],
        z_planes=params['z_planes'],
        smoothing_sigma=params['smoothing_sigma'])

    if params['engine'] == 'ray_propagation':
        optic.set_materials(input_material=base,
                            mixture_materials_list=(base, mixer),
                            output_material=base)
        rays = dict(radius=params['radius'],
                    max_z_angle_deg=params['max_z_angle_deg'],
                    x0=0, y0=0, num_rays=params['num_rays'])
        if params['target'] == 'Beam steering':
            data_source = engine.TrainingData_for_beam_steering(
                coords, steering_angle_deg=params['steering_angle_deg'],
                azimuth_deg=params['azimuth_deg'], **rays)
        else:
            data_source = engine.TrainingData_for_2dImaging(coords, **rays)
        def set_training_pair(optic):
            input_rays, desired_output_rays = (
                data_source.input_output_pair(wavelength))
            optic.set_input_raybundle(input_rays)
            optic.set_desired_output_raybundle(desired_output_rays)
        gradient_update_kwargs['dt'] = params['dt']
    else:
        optic.set_materials((base, mixer))
        data_source = engine.TrainingData_for_2dImaging(
            coords, radius=params['radius'])
        def set_training_pair(optic):
            x0, y0 = data_source.random_point_in_a_circle()
            input_field, desired_output_field = data_source.input_output_pair(
                x0, y0, wavelength, params['divergence_angle_degrees'])
            optic.set_2d_input_field(input_field, wavelength)
            optic.set_2d_desired_output_field(desired_output_field)
        gradient_update_kwargs['momentum'] = params['momentum']

    if params['start'] == 'GRIN lens':
        optic.set_3d_concentration(
            half_pitch_grin_concentration(optic, wavelength))
    elif params['start'] == 'Previous export':
        import tifffile as tf
        concentration = tf.imread(
            Path(params['start_from']) / 'concentration.tif')
        if concentration.shape != (params['n_z'], n_xy, n_xy):
            raise ValueError(
                "That export's grid is %s (z, y, x), but this design's is %s."%(
                    concentration.shape, (params['n_z'], n_xy, n_xy)))
        optic.set_3d_concentration(concentration.astype('float64'))
    return optic, set_training_pair, gradient_update_kwargs

def load_output(output_dir):
    import tifffile as tf
    output_dir = Path(output_dir)
    index = tf.imread(output_dir / 'index.tif')
    metadata = json.loads((output_dir / 'metadata.json').read_text())
    return index, metadata

def plot_slice(image, extent, xlabel, ylabel, vmin, vmax):
    import matplotlib as mpl
    mpl.use('agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    im = ax.imshow(image, extent=extent, origin='lower', vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, label='Refractive index')
    return fig

def main():
    import streamlit as st
    st.set_page_config(page_title="Freeform 3D optical design", layout='wide')
    st.title("Freeform 3D optical design")

    with st.sidebar:
        engine = st.radio("Engine", list(DEFAULTS), key='engine')
        target = 'Inverted image'
        if engine == 'ray_propagation': # The beam engine only images
            target = st.radio("Target", list(TARGETS), key='target')
        d = dict(DEFAULTS[engine], **TARGETS[target])
        def key(name): # Separate widget state per engine, and per target
            # wherever the target changes the default:
            return '_'.join([engine] + [target]*(name in TARGETS[target])
                            + [name])
        def number(label, name, **kwargs):
            return st.number_input(label, value=d[name], key=key(name),
                                   **kwargs)
        with st.form('design'):
            st.subheader("Grid")
            params = dict(
                engine=engine, target=target, z_planes=d['z_planes'],
                half_width=number("XY half-width", 'half_width', min_value=0.1),
                z_length=number("Z length", 'z_length', min_value=0.1),
                n_xy=number("Voxels in x and y", 'n_xy',
                            min_value=21 if engine == 'beam_propagation' else 2),
                n_z=number("Voxels in z", 'n_z', min_value=2))
            st.subheader("Materials")
            params['base_index'] = st.number_input(
                "Base index", value=1.0, min_value=1.0, key='base_index')
            params['mixer_index'] = st.number_input(
                "Mixer index", value=1.5, min_value=1.0, key='mixer_index')
            st.subheader("Optimization")
            params['wavelength_um'] = number(
                "Wavelength (µm)", 'wavelength_um', min_value=0.01)
            params['iterations'] = st.number_input(
                "Iterations", value=30, min_value=1, key='iterations')
            params['step_size'] = number("Step size", 'step_size',
                                         min_value=0.0001)
            params['smoothing_sigma'] = number(
                "Smoothing sigma", 'smoothing_sigma', min_value=0.0)
            if engine == 'beam_propagation':
                params['momentum'] = number(
                    "Momentum", 'momentum', min_value=0.0, max_value=0.99)
            st.subheader("Starting design")
            starts = ('GRIN lens', 'Uniform', 'Previous export')
            params['start'] = st.radio(
                "Start from", starts, key=key('start'),
                index=starts.index(d.get('start', 'GRIN lens')),
                help="A GRIN lens already images the input face onto the "
                "output face, inverted, so it starts much closer to the "
                "best design for the 'Inverted image' target.")
            params['start_from'] = st.text_input(
                "Previous export folder (for 'Previous export')",
                key='start_from')
            st.subheader(target)
            if target == 'Beam steering':
                params['steering_angle_deg'] = number(
                    "Steering angle (deg)", 'steering_angle_deg',
                    min_value=0.0, max_value=89.0)
                params['azimuth_deg'] = number(
                    "Toward azimuth (deg, 0 is +x)", 'azimuth_deg')
            params['radius'] = number("Input radius", 'radius', min_value=0.01)
            if engine == 'ray_propagation':
                params['max_z_angle_deg'] = number(
                    "Max angle (deg)", 'max_z_angle_deg',
                    min_value=0.0, max_value=89.0)
                params['num_rays'] = number("Rays", 'num_rays', min_value=1)
                params['dt'] = number("Ray time step", 'dt', min_value=0.001)
            else:
                params['divergence_angle_degrees'] = number(
                    "Divergence angle (deg)", 'divergence_angle_degrees',
                    min_value=0.1)
            runs_dir = st.text_input(
                "Save runs under", key='runs_dir',
                value=str(OUTPUT_ROOT / 'ui_runs'))
            submitted = st.form_submit_button("Run design", key='run_design')

    if submitted:
        output_dir = str(Path(runs_dir) / time.strftime('%Y%m%d_%H%M%S'))
        try:
            optic, set_training_pair, gradient_update_kwargs = build_run(
                params)
        except (OSError, ValueError) as e:
            st.error("Can't start from that export: %s"%(e))
            st.stop()
        progress = st.progress(0.0, text="Designing...")
        loss_chart = st.empty()
        iterations = params['iterations']
        def show_progress(losses):
            loss_chart.line_chart({'loss': losses})
            progress.progress(len(losses) / iterations,
                              text="Iteration %i of %i"%(
                                  len(losses), iterations))
        design_and_export(optic, set_training_pair,
                          params['wavelength_um'], iterations, output_dir,
                          progress=show_progress, **gradient_update_kwargs)
        progress.empty()
        loss_chart.empty()
        st.session_state['last_run'] = output_dir
        st.session_state['run'] = output_dir # Show the new design

    runs = sorted(str(p.parent) for p in OUTPUT_ROOT.rglob('metadata.json'))
    last_run = st.session_state.get('last_run')
    if last_run and last_run not in runs:
        runs.append(last_run)
    if not runs:
        st.info("Run a design, or put an exported design under %s."%(
            OUTPUT_ROOT))
        return
    run = st.selectbox("Design to view", runs, key='run')
    index, metadata = load_output(run)

    nz, ny, nx = index.shape
    (x0, y0, z0), (dx, dy, dz) = (metadata['first_voxel_center_xyz'],
                                  metadata['voxel_size_xyz'])
    x_extent = (x0 - dx/2, x0 + (nx - 0.5)*dx)
    y_extent = (y0 - dy/2, y0 + (ny - 0.5)*dy)
    z_extent = (z0 - dz/2, z0 + (nz - 0.5)*dz)
    vmin, vmax = metadata['index_min'], metadata['index_max']

    left, right = st.columns(2)
    with left:
        which_z = st.slider("XY slice at z index", 0, nz - 1, nz//2)
        st.pyplot(plot_slice(index[which_z], x_extent + y_extent,
                             'x', 'y', vmin, vmax))
    with right:
        st.caption("XZ cross-section at the middle of y")
        st.pyplot(plot_slice(index[:, ny//2, :].T, z_extent + x_extent,
                             'z', 'x', vmin, vmax))
    if 'held_out_loss' in metadata:
        st.metric("Held-out loss", "%.4g"%(metadata['held_out_loss']),
                  help="Mean loss over a fixed set of training pairs, so "
                  "it's comparable between designs.")
    st.subheader("Loss history")
    st.line_chart({'loss': metadata.get('loss_history', [])})
    st.subheader("Metadata")
    st.json(metadata)
    for name in ('index.tif', 'concentration.tif', 'metadata.json'):
        st.download_button("Download " + name,
                           (Path(run) / name).read_bytes(), file_name=name)

if __name__ == '__main__':
    main()
