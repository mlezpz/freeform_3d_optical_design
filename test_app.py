import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app import DEFAULTS, TARGETS, build_run, load_output
from end_to_end import export_index_volume

def tiny_params(engine):
    params = dict(DEFAULTS[engine], engine=engine, base_index=1.0,
                  mixer_index=1.5, iterations=2, start='GRIN lens',
                  start_from='', target='Inverted image')
    params.update(n_xy=21, n_z=11, num_rays=50)
    return params

@pytest.mark.parametrize('engine', list(DEFAULTS))
def test_build_run_makes_a_trainable_optic(engine):
    optic, set_training_pair, gradient_update_kwargs = build_run(
        tiny_params(engine))

    assert type(optic).__module__ == engine
    assert tuple(map(int, optic.coordinates.n_xyz)) == (21, 21, 11)
    assert [m.index for m in optic.material_list] == [1.0, 1.5]
    set_training_pair(optic)
    optic.gradient_update(**gradient_update_kwargs)
    assert np.isfinite(float(optic.loss))

def test_load_output_round_trips_an_export(tmp_path):
    optic, _, _ = build_run(tiny_params('ray_propagation'))
    exported_index = export_index_volume(optic, 0.5, tmp_path)

    index, metadata = load_output(tmp_path)

    np.testing.assert_allclose(index, exported_index, rtol=1e-6)
    assert metadata['shape_zyx'] == list(index.shape)

@pytest.mark.parametrize('engine', list(DEFAULTS))
def test_app_runs_a_design_and_shows_it(engine, tmp_path):
    at = AppTest.from_file('app.py', default_timeout=120).run()
    at.radio(key='engine').set_value(engine).run()
    at.number_input(key=engine + '_n_xy').set_value(21)
    at.number_input(key=engine + '_n_z').set_value(11)
    if engine == 'ray_propagation':
        at.number_input(key=engine + '_num_rays').set_value(50)
    at.number_input(key='iterations').set_value(2)
    at.text_input(key='runs_dir').set_value(str(tmp_path))
    at.button(key='run_design').click().run()

    assert not at.exception
    run_dir = at.session_state['last_run']
    assert run_dir.startswith(str(tmp_path))
    assert at.selectbox(key='run').value == run_dir
    shown_metadata = at.json[0].value
    assert '"engine": "%s"'%engine in shown_metadata
    assert '"iterations": 2' in shown_metadata
    assert at.metric[0].label == "Held-out loss"

def test_build_run_can_start_from_a_previous_export(tmp_path):
    optic, _, _ = build_run(tiny_params('ray_propagation'))
    export_index_volume(optic, 0.5, tmp_path)
    exported = optic.concentration

    params = dict(tiny_params('ray_propagation'), start='Previous export',
                  start_from=str(tmp_path))
    restarted, _, _ = build_run(params)

    np.testing.assert_allclose(restarted.concentration, exported, rtol=1e-6)
    with pytest.raises(ValueError, match="grid"):
        build_run(dict(params, n_z=12))

def test_build_run_makes_a_beam_steering_run():
    params = dict(tiny_params('ray_propagation'), target='Beam steering',
                  **TARGETS['Beam steering'])
    optic, set_training_pair, gradient_update_kwargs = build_run(params)

    assert gradient_update_kwargs['z_planes'] == (20, 40)
    assert gradient_update_kwargs['step_size'] == 0.1
    set_training_pair(optic)
    v = optic.desired_output_raybundle.v_xyz
    np.testing.assert_allclose(v[0] / v[2], np.tan(np.deg2rad(10)))
    np.testing.assert_allclose(v[1], 0, atol=1e-12)

def test_app_runs_a_beam_steering_design(tmp_path):
    at = AppTest.from_file('app.py', default_timeout=120).run()
    at.radio(key='engine').set_value('ray_propagation').run()
    at.radio(key='target').set_value('Beam steering').run()
    at.number_input(key='ray_propagation_n_xy').set_value(21)
    at.number_input(key='ray_propagation_n_z').set_value(11)
    at.number_input(key='ray_propagation_num_rays').set_value(50)
    at.number_input(key='iterations').set_value(2)
    at.text_input(key='runs_dir').set_value(str(tmp_path))
    at.button(key='run_design').click().run()

    assert not at.exception
    assert at.radio(key='ray_propagation_Beam steering_start').value == 'Uniform'
    shown_metadata = at.json[0].value
    assert '"step_size": 0.1' in shown_metadata
    assert '"z_planes": [20, 40]' in shown_metadata
