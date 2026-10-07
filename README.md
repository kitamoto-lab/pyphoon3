# pyphoon3

<!-- ABOUT THE PROJECT -->
## About The Project

The Digital Typhoon Project is a project aimed to be an example of the application of meteoinformatics to large-scale 
real-world issues. The two primary challenges of this project are to (1) build, for the typhoon image collection, 
large-scale scientific databases which are the foundation of meteoinformatics, and (2) to establish algorithms 
and database models for the discovery of information and knowledge useful for typhoon analysis and prediction. 

pyphoon3 is the PyTorch dataloader for the **Digital Typhoon Dataset V3**. Through it the user can
(1) access typhoon images via index, typhoon ID, or season, (2) load single-channel or multi-channel data,
(3) load all data into memory if desired, and (4) split the dataset for model training by image, sequence (typhoon),
or season while preventing leakage between the splits.

Related information is available at https://github.com/kitamoto-lab/digital-typhoon/.

<!-- DATASET -->
## Digital Typhoon Dataset V3

The dataset is available at http://agora.ex.nii.ac.jp/digital-typhoon/dataset/ under the
Creative Commons Attribution 4.0 International (CC BY 4.0) license.
The current release, **V3-2025**, includes data up to the 2025 season.

V3 covers two basins, **WP** (Western Pacific, Northern Hemisphere) and **AU** (around Australia, Southern Hemisphere),
and four satellite channels:

| Channel | Type |
|---|---|
| 1 | Infrared 1 (the channel of datasets V1 and V2, longest temporal coverage) |
| 2 | Infrared 2 |
| 3 | Water vapor |
| 4 | Near infrared |

The data is organized first by channel, then by basin:

```
V3-2025/
├── 1/
│   ├── WP/
│   │   ├── image/<sequence ID>/<YYYYMMDDHH>-<sequence ID>-<satellite>-1.h5
│   │   ├── metadata/<sequence ID>.csv
│   │   └── metadata.json
│   └── AU/ ...
├── 2/ ...
├── 3/ ...
└── 4/ ...
```

Each h5 file holds one 512 × 512 image in the dataset `Data`.

<!-- VERSIONS -->
## Versions

| pyphoon3 | Dataset | Notes |
|---|---|---|
| 3.0.0 | V3-2025 | Used for the benchmarks of the Digital Typhoon Dataset V3 paper |

Releases 3.0.x only contain fixes that leave the paper's benchmark results unchanged, so that the results stay
reproducible. Changes that may affect the results are made in a new minor version.

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

This project uses:
* python3
* torch
* torchvision
* numpy
* pandas
* h5py
* psutil

### Installation

1. Clone and enter the repo 
    ```sh
    git clone https://github.com/kitamoto-lab/pyphoon3
    cd pyphoon3
    ```
2. Install the package
    ```sh
    pip3 install .
    ```
3. To uninstall, run
    ```sh
    pip3 uninstall pyphoon3
    ```

### Sample data

To try pyphoon3 without downloading the full dataset (about 240 GB), use the sample dataset (37 MB).
It has the same directory layout as the full dataset and contains channel 1 of 4 typhoons per basin
(2 in the 2022–2023 seasons and 1 in each of the 2024 and 2025 seasons), thinned to 6-hourly observations.

```sh
wget https://github.com/kitamoto-lab/pyphoon3/releases/download/sample-V3-2025/pyphoon3-sample-V3-2025.zip
unzip pyphoon3-sample-V3-2025.zip
```

The sample is created from the full dataset with `tools/make_sample_dataset.py`
(`--channels 1 2 3 4` creates a sample with all channels, for trying multi-channel loading).

### Usage

1. Import the Dataset class
    ```python
    from pyphoon3.DigitalTyphoonDataset import DigitalTyphoonDataset
    ```
   You can also import the submodules `DigitalTyphoonSequence`, `DigitalTyphoonImage`, and 
    `DigitalTyphoonUtils` in the same way if desired:
    ```python
    from pyphoon3.DigitalTyphoonSequence import DigitalTyphoonSequence
    from pyphoon3.DigitalTyphoonImage import DigitalTyphoonImage
    from pyphoon3.DigitalTyphoonUtils import *
    ```
2. Instantiate the loader for one channel and one basin (here channel 1, WP)
    ```python
    root = "pyphoon3-sample-V3-2025"   # or the full dataset, e.g. "/path/to/V3-2025"
    dataset_obj = DigitalTyphoonDataset(f"{root}/1/WP/image/",
                                        f"{root}/1/WP/metadata/",
                                        f"{root}/1/WP/metadata.json",
                                        'pressure',  # label to return when indexing
                                        split_dataset_by='sequence',
                                        load_data_into_memory=False,
                                        ignore_list=[],
                                        verbose=False)
    ```
   Available labels: `year`, `month`, `day`, `hour`, `grade`, `lat`, `lng`, `pressure`, `wind`,
   `dir50`, `long50`, `short50`, `dir30`, `long30`, `short30`, `landfall`, `interpolated`.
   Images whose label is missing (`-1`) are skipped when loading.
3. Or instantiate it for several channels (needs the full dataset or a sample with more channels).
   Only observation times where all the given channels exist are kept,
   and each image has the shape `(number of channels, 512, 512)`.
    ```python
    channels = [1, 2, 3]
    dataset_obj = DigitalTyphoonDataset("", "", "", 'pressure',
                                        image_dirs=[f"{root}/{c}/WP/image/" for c in channels],
                                        metadata_dirs=[f"{root}/{c}/WP/metadata/" for c in channels],
                                        metadata_jsons=[f"{root}/{c}/WP/metadata.json" for c in channels])
    ```

The dataset object is now instantiated and you can use the data in the desired fashion. Some examples include: 

* Get the length of the dataset
    ```python
    length = len(dataset_obj)
    ```
  
* Get the item at the i'th index
    ```python
    image_array, label = dataset_obj[i]  # label corresponds to the label passed in on instantiation or set via dataset_obj.set_label()
    image_obj = dataset_obj.get_image_from_idx(i)
    image_obj.image()  # Get the image pixels in a numpy array
    image_obj.year()   # Get the year the image was taken
    image_obj.grade()  # Get the grade of the typhoon at the time of the image
    ```

* Split the dataset by season: the last two seasons for evaluation and all earlier seasons for training
  (the SS-L2 split used for the V3 benchmarks)
    ```python
    eval_set = dataset_obj.images_from_seasons([2024, 2025])
    train_set = dataset_obj.images_from_seasons([s for s in dataset_obj.get_seasons() if s < 2024])
    ```

* Split the dataset randomly into train, test, and validation sets
    ```python
    from torch.utils.data import DataLoader

    train, test, val = dataset_obj.random_split([0.7, 0.15, 0.15], split_by='sequence')
    trainloader = DataLoader(train, batch_size=16, shuffle=True)
    ```

### Images without usable data

Some images in channels 1–3 of V3-2025 have no usable data: they are blank (a single value, or two) or mostly
missing (more than half of the pixels at the fill value). pyphoon3 flags such images; they are kept by default.

* List the flagged images (checked on the first call, which reads the file sizes of all images and opens only the
  small files; this takes some time on a network file system)
    ```python
    indices = dataset_obj.get_flagged_indices()  # dataset indices
    files = dataset_obj.get_flagged_files()      # file paths
    ```
* Skip them when loading; with several channels, an observation is skipped if any of its channels is flagged
    ```python
    dataset_obj = DigitalTyphoonDataset(..., exclude_flagged=True)
    ```

The benchmarks of the V3 paper keep all images (the default). None of the flagged images are in the 2024–2025
evaluation seasons.

<!-- EXAMPLES -->
## Examples

`examples/train_regression.py` trains a ResNet to regress typhoon intensity (central pressure or maximum wind)
from satellite images with the benchmark protocol of the V3 paper: ResNet-18 or ResNet-50 with ImageNet weights,
MSE loss, Adam (learning rate 1e-4), batch size 16, 50 epochs, and the SS-L2 split
(2024–2025 for evaluation, earlier seasons for training). Images are clipped to [170, 300] and scaled to [0, 1].

Try it on the sample data (a smoke test; the sample is far too small to train a meaningful model):
```sh
python examples/train_regression.py --data-root pyphoon3-sample-V3-2025 --basin WP --channels 1 --epochs 2
```

Benchmark on the full dataset, e.g. WP, channel 1, pressure, ResNet-18, full 512 × 512 images
(the paper reports an evaluation RMSE of 6.56 ± 0.06 hPa over 5 runs):
```sh
python examples/train_regression.py --data-root /path/to/V3-2025 --basin WP --channels 1 \
    --label pressure --model resnet18 --image-type full --seed 0
```

Other options: `--basin AU`, `--channels 1 2 3` (multi-channel), `--label wind`, `--model resnet50`,
`--image-type resized` or `cropped` (224 × 224). The full WP channel 1 dataset reads about 200 GB per epoch,
so store the dataset on a fast local disk.

<!-- TESTS -->
## Tests

The tests run on the sample data. Unzip it into `tests/` (or set `PYPHOON3_SAMPLE_DIR`) and run from the repository root:
```sh
python -m unittest discover -s tests -t tests
```
See `tests/README.md` for details.

<!-- CITATION -->
## Citation

Please cite the dataset as:

> Digital Typhoon Dataset V3 (National Institute of Informatics) doi: https://doi.org/10.20783/DIAS.664

<!-- TODO: add the citation of the V3 paper after the arXiv submission -->
