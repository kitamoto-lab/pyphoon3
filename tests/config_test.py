"""
Shared configuration for the tests: location of the sample dataset and helpers to read it directly.

The tests use the pyphoon3 sample of the Digital Typhoon Dataset V3 (see README.md, "Sample data").
Unzip it into tests/ or point PYPHOON3_SAMPLE_DIR to it. The sample has channel 1 only; the multi-channel tests run
when the sample is created with more channels (tools/make_sample_dataset.py --channels 1 2 3 4).
"""
import csv
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SAMPLE_DIR = os.environ.get('PYPHOON3_SAMPLE_DIR',
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pyphoon3-sample-V3-2025'))
CHANNELS = [1, 2, 3, 4]
BASINS = ['WP', 'AU']
EVAL_SEASONS = [2024, 2025]


def require_sample_data():
    if not os.path.isdir(SAMPLE_DIR):
        raise unittest.SkipTest(f'sample dataset not found at {SAMPLE_DIR}; '
                                f'download it (see README.md) or set PYPHOON3_SAMPLE_DIR')


def has_channels(channels, basin='WP'):
    return all(os.path.isdir(data_dir(c, basin)) for c in channels)


def data_dir(channel, basin):
    return os.path.join(SAMPLE_DIR, str(channel), basin)


def dataset_args(channel, basin):
    """(image_dir, metadata_dir, metadata_json) for DigitalTyphoonDataset."""
    d = data_dir(channel, basin)
    return d + '/image/', d + '/metadata/', d + '/metadata.json'


def multi_dataset_kwargs(channels, basin):
    """image_dirs / metadata_dirs / metadata_jsons keyword arguments for a multi-channel DigitalTyphoonDataset."""
    return dict(image_dirs=[data_dir(c, basin) + '/image/' for c in channels],
                metadata_dirs=[data_dir(c, basin) + '/metadata/' for c in channels],
                metadata_jsons=[data_dir(c, basin) + '/metadata.json' for c in channels])


def image_files(channel, basin):
    """{sequence ID: sorted list of h5 filenames} read directly from disk."""
    root = data_dir(channel, basin) + '/image'
    return {seq: sorted(f for f in os.listdir(f'{root}/{seq}') if f.endswith('.h5')) for seq in sorted(os.listdir(root))}


def track_rows(channel, basin, seq):
    """{h5 filename: track row (dict of strings)} read directly from the track CSV."""
    with open(f'{data_dir(channel, basin)}/metadata/{seq}.csv', newline='') as f:
        return {r[f'file_{channel}']: r for r in csv.DictReader(f) if r[f'file_{channel}']}
