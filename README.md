# EPM: Embedding-aware Polarization Management

This repository provides the preprocessing code for experiments on
signed networks used in the accompanying paper.

## Datasets

All datasets used in this work are publicly available signed networks.
Raw data files are not included in this repository.

Please refer to `data/README.md` for dataset sources and download instructions.

## Preprocessing

Preprocessing can be executed using the provided script:

```bash
bash scripts/run_preprocessing.sh
```

Supported datasets:
bitcoinalpha, bitcoinotc, wiki-RfA, wiki-Elec, Slashdot, Epinions.

## Training and Measurement

After preprocessing, training and polarization measurement can be run with:

```bash
bash scripts/run_measurement.sh
```
By default, the script runs on the bitcoinalpha dataset using the default training settings
and computes polarization with neg_scale = 0.1.

Optional arguments:
```bash
bash scripts/run_measurement.sh --datasets "bitcoinalpha bitcoinotc" --neg-scale 0.2
```

## Mitigation Preparation

To prepare gray-zone information for mitigation (community clustering, PCS pairs, and gray-node scores), run:

```bash
bash scripts/run_mitigation_prep.sh
```

By default, the script runs on the bitcoinalpha dataset with seed 0.
Large datasets (e.g., Slashdot, Epinions) are handled automatically using the large-scale PCS pipeline.


## Mitigation

After preprocessing and base measurement, gray-zone–based mitigation can be executed with:

```bash
bash scripts/run_mitigation.sh
```

By default, the script runs a single representative mitigation setting on the bitcoinalpha dataset
with seed 0, and summarizes the results together with a comparison against the base model.

To run mitigation on other datasets or multiple seeds, use the optional arguments:
```bash
bash scripts/run_mitigation.sh --datasets "bitcoinalpha bitcoinotc" --seeds "0 1"
```




