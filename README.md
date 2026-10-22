# RTED-Net

Official code and selected checkpoints for **RTED-Net: Reliability-Guided Temporal Evidence Learning for Multi-Frame Infrared Small Target Detection**.

RTED-Net segments the center frame of a seven-frame grayscale infrared sequence. It uses ordered temporal cues, a spatial reliability estimator, reliability-guided fusion, frequency-aware temporal memory, and bounded recovery and suppression of the coarse prediction.

## Contents

| Path | Description |
| --- | --- |
| `model.py` | RTED-Net implementation (`TFGNetV101` class name retained for checkpoint compatibility) |
| `datasets.py` | NUDT-MIRSDT and generic sequence loaders |
| `train.py` | Training and validation |
| `evaluate.py` | Segmentation and detection metrics |
| `weights/` | Selected NUDT-MIRSDT and IRDST checkpoints |
| `scripts/` | Example training and evaluation commands |

The repository does not redistribute the datasets. Download them from their respective dataset providers and follow their terms of use. The model weights are provided as research artifacts; dataset and third-party software licenses remain with their original owners.

## Setup

Python 3.10 or newer is recommended. Install a PyTorch build suitable for your CUDA setup from the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The local smoke test used Python 3.13.5 and PyTorch 2.11 CPU. Training commands below assume a CUDA GPU.

## Data layout

For NUDT-MIRSDT, the loader expects this layout and split files supplied with the dataset:

```text
NUDT-MIRSDT/
  train.txt     # entries such as Sequence1/Mix/00001.mat
  test.txt
  Sequence1/
    images/00001.png
    masks/00001.png
  Sequence2/
    images/...
    masks/...
```

The generic sequence loader used for IRDST expects sequence directories containing `images/` and `masks/`. Its split files can list sequence names or pairs of sequence names and frame stems. Pass the matching train and test split files with `--train-split` and `--val-split`. Run `python train.py --help` and `python evaluate.py --help` for all options.

## Evaluation

Set `DATA_ROOT` to the extracted dataset directory. For example:

```bash
python evaluate.py \
  --data-root /path/to/NUDT-MIRSDT \
  --val-split /path/to/NUDT-MIRSDT/test.txt \
  --checkpoint weights/nudt_v101_7f_native_best.pt \
  --output-dir runs/nudt_eval \
  --num-frames 7 --base-channels 24 \
  --input-size 512x512 --threshold 0.5 \
  --device cuda
```

For IRDST, use `weights/irdst_v101_7f_384x256_real_best.pt`, `--input-size 384x256`, and `--allow-resize-larger` with the matching IRDST paths. Example scripts are in `scripts/`.

`evaluate.py` writes metric summaries and per-image results under `--output-dir`. Its component false-alarm metrics and pixel false-alarm metrics are distinct; inspect the reported metric keys and the paper's protocol before comparing numbers.

## Training

Training without a checkpoint begins from random initialization. Example:

```bash
python train.py \
  --data-root /path/to/NUDT-MIRSDT \
  --train-split /path/to/NUDT-MIRSDT/train.txt \
  --val-split /path/to/NUDT-MIRSDT/test.txt \
  --output-dir runs/nudt_train \
  --num-frames 7 --pad-size 512x512 \
  --base-channels 24 --epochs 80 \
  --batch-size 1 --val-batch-size 1
```

The scripts reproduce saved experiment configurations and may initialize from a supplied checkpoint. Adjust paths, batch size, workers, and precision for your hardware. Datasets and full training runs are needed to verify the paper's reported numbers; the included smoke test checks code execution only.

## Checkpoints

| File | Intended dataset | Size |
| --- | --- | ---: |
| `weights/nudt_v101_7f_native_best.pt` | NUDT-MIRSDT | 34.6 MiB |
| `weights/irdst_v101_7f_384x256_real_best.pt` | IRDST real | 34.6 MiB |

Load only checkpoints from trusted sources, as PyTorch checkpoint formats may contain executable pickle data.

## Citation

The accompanying manuscript is **RTED-Net: Reliability-Guided Temporal Evidence Learning for Multi-Frame Infrared Small Target Detection**. A formal citation will be added when publication metadata is available.

## License

The code and the two included model checkpoints are released under the [MIT License](LICENSE). Dataset downloads are separate and subject to their own terms.
