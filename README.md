# EPM: Embedding-aware Polarization Management

This repository provides the implementation for preprocessing, signed network embedding training, embedding-aware polarization measurement, and gray-zone-based mitigation used in the accompanying paper.

The default scripts are configured for a lightweight reviewer-friendly run on `bitcoinalpha` with seed `0`. Full multi-seed and grid-search experiments can be run by passing additional command-line options.

## Datasets

All datasets used in this work are publicly available signed networks. Raw data files are not included in this repository.

Please refer to `data/README.md` for dataset sources and download instructions.

Supported datasets:

- `bitcoinalpha`
- `bitcoinotc`
- `wiki-RfA`
- `wiki-Elec`
- `Slashdot`
- `Epinions`

## Quick Start

After placing the raw files under `data/raw`, run the default pipeline:

```bash
bash scripts/run_preprocessing.sh
bash scripts/run_measurement.sh
bash scripts/run_mitigation_prep.sh
bash scripts/run_mitigation.sh
```

By default, the scripts run on `bitcoinalpha` with seed `0`.

## Preprocessing

```bash
bash scripts/run_preprocessing.sh
```

This step converts raw signed edge files into a common format, builds a static signed graph snapshot, creates train/validation/test splits, and estimates the number of communities.

To preprocess other datasets:

```bash
bash scripts/run_preprocessing.sh bitcoinotc wiki-Elec wiki-RfA
bash scripts/run_preprocessing.sh --dataset Slashdot
```

## Base Training and Measurement

```bash
bash scripts/run_measurement.sh
```

The default setting is:

```text
dataset       = bitcoinalpha
seed          = 0
embedding_dim = 128
num_layers    = 3
lr            = 0.001
neg_scale     = 0.1
```

To run multiple datasets or seeds:

```bash
bash scripts/run_measurement.sh --datasets "bitcoinalpha bitcoinotc"
bash scripts/run_measurement.sh --datasets "bitcoinalpha" --seeds "0 1 2 3 4"
```

To run a hyperparameter grid search:

```bash
bash scripts/run_measurement.sh \
  --embedding-dims "32 64 128" \
  --num-layers "2 3 4" \
  --lrs "0.05 0.01 0.005 0.001 0.0005"
```

Main outputs are saved under `results/base/`.

## Mitigation Preparation

```bash
bash scripts/run_mitigation_prep.sh
```

This step prepares KMeans communities, PCS-based polarized community pairs, and gray-node scores.

Large datasets such as `Slashdot` and `Epinions` are automatically handled using the large-scale PCS pipeline.

```bash
bash scripts/run_mitigation_prep.sh --datasets "Slashdot Epinions"
```

## EPM Mitigation

```bash
bash scripts/run_mitigation.sh
```

The default mitigation setting is:

```text
dataset       = bitcoinalpha
seed          = 0
tau           = 0.5
d_max         = 3
gamma         = 1.5
embedding_dim = 128
num_layers    = 4
lr            = 0.001
neg_scale     = 0.1
```

This step performs gray-zone edge augmentation, retrains the signed embedding model, computes polarization after mitigation, and compares the result with the base model.

To run multiple datasets or seeds:

```bash
bash scripts/run_mitigation.sh --datasets "bitcoinalpha bitcoinotc" --seeds "0 1"
```

To search over mitigation hyperparameters:

```bash
bash scripts/run_mitigation.sh \
  --tau "0.5 0.6 0.7 0.8 0.9" \
  --d-max "2 3" \
  --gamma "0.5 1.0 1.5"
```

Main outputs are saved under `results/gray/`.

## Notes

- The default scripts run a single representative configuration.
- The paper experiments used multiple seeds and hyperparameter grid searches.
- Validation and test edges are used only for evaluation; message passing uses training edges only.
- Polarization measurement uses PCA-based opinion coordinates, node-wise L2 normalization, and a solver-based Laplacian computation.
- Raw dataset files and generated result files are not included in this repository.
