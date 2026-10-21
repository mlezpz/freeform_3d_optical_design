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
  record (iterations, `gradient_update` arguments, loss history).

To export an optic you've already designed, without optimizing it again, call
`export_index_volume(optic, wavelength_um, output_dir)`.

Tests: `pytest test_end_to_end.py`.
