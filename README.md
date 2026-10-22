# freeform_3d_optical_design

## From design spec to fabrication-ready volume

`end_to_end.py` runs the optimization for a `Refractive3dOptic` from either
engine and exports the result. Run `python end_to_end.py` for a small
ray-optics demo, or call it yourself:

```python
from end_to_end import design_and_export

loss_history = design_and_export(
    optic, set_training_pair, wavelength_um=0.5, iterations=1000,
    output_dir='output/my_design',
    **gradient_update_kwargs) # Passed to optic.gradient_update()
```

`set_training_pair(optic)` is called every iteration to set the input and
desired output. For `ray_propagation`:

```python
def set_training_pair(optic):
    input_rays, desired_output_rays = data_source.input_output_pair(0.5)
    optic.set_input_raybundle(input_rays)
    optic.set_desired_output_raybundle(desired_output_rays)
```

For `beam_propagation`:

```python
def set_training_pair(optic):
    x0, y0 = data_source.random_point_in_a_circle()
    input_field, desired_output_field = data_source.input_output_pair(
        x0, y0, wavelength=1, divergence_angle_degrees=15)
    optic.set_2d_input_field(input_field, 1)
    optic.set_2d_desired_output_field(desired_output_field)
```

The output directory gets three files:

- `index.tif`: the refractive index at each voxel at `wavelength_um`, as a
  float32 ZYX stack with the voxel size embedded, so ImageJ shows physical
  dimensions.
- `concentration.tif`: the 0–1 mixture fraction at each voxel. It doesn't
  depend on wavelength, and you can load it back with `set_3d_concentration`.
- `metadata.json`: the engine, axis order, shape, first voxel center, voxel
  size (in the units of your `Coordinates`; microns if you use
  `SellmeierMaterial`), wavelength, index range, materials, and the run
  record (iterations, `gradient_update` arguments, loss history, and the
  held-out loss).

Training loss is noisy, because each iteration uses a different random
training pair. `held_out_loss(optic, set_training_pair, **gradient_update_kwargs)`
scores a design on a fixed, seeded set of pairs instead, so you can compare
designs. That's the `held_out_loss` recorded in the metadata.

## Starting design

Where the optimization starts matters more than how long it runs. For the
built-in imaging target, start from a graded-index lens that already images
the input face onto the output face:

```python
from end_to_end import half_pitch_grin_concentration
optic.set_3d_concentration(half_pitch_grin_concentration(optic, wavelength_um))
```

On the UI's default grids, this lens scores better before any training than
2000 iterations from a uniform start do (held-out loss 0.052 vs 0.067 for
the ray engine, 0.012 vs about 0.023 for the beam engine). 500 iterations
from it reach 0.049 and 0.011. To continue an earlier
design instead, load its `concentration.tif` with `set_3d_concentration`.

To export an optic you've already designed, without optimizing it again, call
`export_index_volume(optic, wavelength_um, output_dir)`.

## Beam-steering target

`ray_propagation.TrainingData_for_beam_steering` asks the optic to tilt
every ray by `steering_angle_deg` toward `azimuth_deg`, like a prism. Each
target ray leaves the output face where the input ray would exit if the
optic were made of the input material, then tilted:

```python
data_source = TrainingData_for_beam_steering(
    coords, radius=3, max_z_angle_deg=0, x0=0, y0=0, num_rays=1000,
    steering_angle_deg=10)
```

Steering light sideways within the optic shifts it sideways too, so
measure the loss in the far field, where the angle dominates:
`z_planes=(20, 40)` rather than the imaging default `(0, 1)`. With the
exit-plane default, the best score comes from not steering at all. Use a
smaller step than for imaging (`step_size=0.1` on the default grid), and
start from a uniform mixture: the GRIN start is only for the imaging
target. On the default grid, 3000 iterations steer collimated light by
8.4° ± 0.5° toward a 10° target, and the steering angle is still slowly
increasing at that point.

## UI

```
pip install streamlit
streamlit run app.py
```

Set up a design for the built-in 2D imaging target in the sidebar: engine,
grid, materials, optimization settings, starting design (a GRIN lens by
default, a uniform mixture, or a previous export) and target settings. Press "Run
design" to watch the loss as it trains. Each run is exported to its own
timestamped folder. Then view it, or any earlier export under `output/`:
XY slices and an XZ cross-section of the index volume (in physical units),
the held-out loss, the loss history, the metadata, and downloads of all
three files.

Tests: `pytest`.
