# SMPL to MimicKit Motion Converter

This tool converts SMPL motion data (specifically from the AMASS dataset) into the MimicKit format compatible with the SMPL humanoid configuration.

## Overview

The converter takes an SMPL `.npz` file containing pose parameters and root translations, retargets it if necessary, applies coordinate conversions, and saves it as a MimicKit-compatible motion file (Pickle format).

## Usage

This script is designed to be run from the **root directory** of the MimicKit repository.

### Command Line Interface

```bash
python tools/smpl_to_mimickit/smpl_to_mimickit.py \
    --input_file <path_to_input.npz> \
    --output_file <path_to_output.pkl> \
    [optional_arguments]
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_file` | str | **Required** | Path to the input SMPL motion file (`.npz` files supported). |
| `--output_file` | str | **Required** | Path where the converted MimicKit motion file will be saved. |
| `--loop` | str | `wrap` | Loop mode for the motion. Options: `wrap`, `clamp`. |
| `--start_frame` | int | `0` | Start frame for clipping the motion. |
| `--end_frame` | int | `-1` | End frame for clipping the motion. `-1` uses all frames up to the end. |
| `--output_fps` | int | `-1` | Target frame rate for the output motion. `-1` keeps the source FPS. |
| `--z_correction` | str | `calibrate` | Method to correct Z-axis (height). Options: `none`, `calibrate` (adjusts based on first 30 frames), `full` (adjusts based on entire motion). |

## Input Format Details

The input file should be a standard AMASS `.npz` file containing the following keys:
*   `poses`: Array of pose parameters with shape `(num_frames, num_pose_params)`.
*   `trans`: Array of root translations with shape `(num_frames, 3)`.
*   `mocap_framerate` or `fps`: Frame rate of the motion (integer or scalar).

## Example

To convert a walking motion from AMASS, correcting for ground height using the full motion:

```bash
python tools/smpl_to_mimickit/smpl_to_mimickit.py \
    --input_file data/amass_sample/walk.npz \
    --output_file motions/smpl_walk.pkl \
    --z_correction full \
    --loop wrap
```

## Credits

Part of the conversion logic is adapted from [PHC](https://github.com/ZhengyiLuo/PHC).