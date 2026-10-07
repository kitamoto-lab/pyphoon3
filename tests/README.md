# Tests for pyphoon3

The tests run on the pyphoon3 sample of the Digital Typhoon Dataset V3 (37 MB, channel 1), which has the same
directory layout as the full dataset. Tests that need the data are skipped when the sample is not found, and the
multi-channel tests are skipped unless the sample has channels 1–3
(create one with `tools/make_sample_dataset.py --channels 1 2 3 4`).

## Setup

Download and unzip the sample dataset into this folder (see "Sample data" in the top-level README.md):

```sh
cd tests
wget https://github.com/kitamoto-lab/pyphoon3/releases/download/sample-V3-2025/pyphoon3-sample-V3-2025.zip
unzip pyphoon3-sample-V3-2025.zip
```

or point `PYPHOON3_SAMPLE_DIR` to an existing copy:

```sh
export PYPHOON3_SAMPLE_DIR=/path/to/pyphoon3-sample-V3-2025
```

The sample is created from the full dataset with `tools/make_sample_dataset.py`.

## Run

From the repository root:

```sh
python -m unittest discover -s tests -t tests
```

A single test module or test:

```sh
python -m unittest discover -s tests -t tests -p test_DigitalTyphoonDataset.py
python -m unittest -k test_season_split discover -s tests -t tests
```

## Contents

- `config_test.py`: location of the sample dataset and helpers that read it directly (image files, track CSVs)
- `test_DigitalTyphoonDataset.py`: single-channel dataset: indexing, labels, chronological order, seasons and the
  SS-L2 split, random splits without leakage, ignore list, filter and transform functions, loading into memory,
  sequence mode, DataLoader
- `test_DigitalTyphoonDataset_MultiChannel.py`: multi-channel dataset: intersection of channels, channel order,
  sequences registered once, season subsets without duplicates, random splits by season and sequence
- `test_DigitalTyphoonImage.py`: image values and track data of one image
- `test_DigitalTyphoonSequence.py`: images and track data of one sequence
- `test_DigitalTyphoonUtils.py`: filename parsing and enums (no data needed)
