# Datasets

All datasets used in this work are publicly available signed networks from the Stanford Network Analysis Project (SNAP):

https://snap.stanford.edu/data/index.html

We use the following signed network datasets:

- `soc-sign-bitcoin-alpha`
- `soc-sign-bitcoin-otc`
- `wiki-Elec`
- `wiki-RfA`
- `soc-sign-Slashdot090221`
- `soc-sign-epinions`

## Included Raw File

For convenience, this repository includes the raw file for the default lightweight run:

```text
data/raw/soc-sign-bitcoinalpha.csv
```

This allows users to run the default pipeline without downloading all datasets.

## Downloading Other Datasets

Other raw dataset files are not included. To run experiments on additional datasets, please download the corresponding raw files from SNAP and place them under `data/raw/`.

Expected directory structure:

```text
data/raw/
 ├── soc-sign-bitcoinalpha.csv
 ├── soc-sign-bitcoinotc.csv
 ├── wikiElec.ElecBs3.txt
 ├── wiki-RfA.txt
 ├── soc-sign-Slashdot090221.txt
 └── soc-sign-epinions.txt
```

## Notes

The preprocessing code expects the file names shown above. If downloaded files have different names, please rename them accordingly before running the preprocessing scripts.
