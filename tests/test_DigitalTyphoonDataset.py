import contextlib
import io
import os
import shutil
import tempfile
from datetime import datetime
from unittest import TestCase

import h5py
import numpy as np
import torch

from config_test import (BASINS, EVAL_SEASONS, require_sample_data, data_dir, dataset_args, image_files, track_rows)
from pyphoon3.DigitalTyphoonDataset import DigitalTyphoonDataset


def setUpModule():
    require_sample_data()


def make_dataset(basin='WP', label='pressure', **kwargs):
    return DigitalTyphoonDataset(*dataset_args(1, basin), label, **kwargs)


class TestDigitalTyphoonDataset(TestCase):
    """Single-channel dataset (channel 1) on the sample data."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = make_dataset()
        cls.files = image_files(1, 'WP')

    def test_len_matches_image_files(self):
        for basin in BASINS:
            with self.subTest(basin=basin):
                dataset = self.dataset if basin == 'WP' else make_dataset(basin)
                self.assertEqual(len(dataset), sum(len(f) for f in image_files(1, basin).values()))

    def test_getitem_returns_image_and_label(self):
        image, label = self.dataset[0]
        self.assertEqual(image.shape, (512, 512))
        self.assertEqual(image.dtype, np.float32)
        self.assertTrue(np.isfinite(image).all())
        image_obj = self.dataset.get_image_from_idx(0)
        row = track_rows(1, 'WP', image_obj.sequence_id())[os.path.basename(image_obj.filepath())]
        self.assertEqual(label, float(row['pressure']))

    def test_labels_match_track_data(self):
        for idx in range(len(self.dataset)):
            image_obj = self.dataset.get_image_from_idx(idx)
            row = track_rows(1, 'WP', image_obj.sequence_id())[os.path.basename(image_obj.filepath())]
            self.assertEqual(image_obj.pressure(), float(row['pressure']))
            self.assertEqual(image_obj.wind(), float(row['wind']))
            self.assertEqual(image_obj.grade(), int(row['grade']))
            self.assertEqual(image_obj.year(), int(row['year']))

    def test_set_label(self):
        dataset = make_dataset()
        dataset.set_label('wind')
        image_obj = dataset.get_image_from_idx(3)
        self.assertEqual(dataset[3][1], image_obj.wind())
        with self.assertRaises(KeyError):
            dataset.set_label('long')  # the longitude label is 'lng'

    def test_multiple_labels(self):
        dataset = make_dataset(label=('pressure', 'wind', 'lat', 'lng'))
        self.assertEqual(len(dataset), len(self.dataset))
        _, labels = dataset[0]
        image_obj = dataset.get_image_from_idx(0)
        np.testing.assert_array_equal(labels, [image_obj.pressure(), image_obj.wind(), image_obj.lat(), image_obj.long()])

    def test_images_are_in_chronological_order_within_sequences(self):
        for seq in self.files:
            images = self.dataset.image_objects_from_sequence(seq)
            self.assertEqual([os.path.basename(i.filepath()) for i in images], self.files[seq])
            times = [i.datetime() for i in images]
            self.assertEqual(times, sorted(times))

    def test_image_datetime_matches_filename(self):
        image_obj = self.dataset.get_image_from_idx(0)
        name = os.path.basename(image_obj.filepath())
        self.assertEqual(image_obj.datetime(), datetime.strptime(name[:10], '%Y%m%d%H'))

    def test_sequences_and_seasons(self):
        self.assertEqual(sorted(self.dataset.get_sequence_ids()), sorted(self.files))
        self.assertEqual(self.dataset.get_number_of_sequences(), len(self.files))
        self.assertEqual(sorted(self.dataset.get_seasons()), [2022, 2023, 2024, 2025])
        for seq, names in self.files.items():
            self.assertTrue(self.dataset.sequence_exists(seq))
            self.assertEqual(len(self.dataset.images_from_sequence(seq)), len(names))
        self.assertFalse(self.dataset.sequence_exists('190001'))

    def test_season_split(self):
        """SS-L2 split used for the V3 benchmarks: train and evaluation sets are disjoint and cover the dataset."""
        eval_idx = self.dataset.images_from_seasons(EVAL_SEASONS).indices
        train_seasons = [s for s in self.dataset.get_seasons() if s not in EVAL_SEASONS]
        train_idx = self.dataset.images_from_seasons(train_seasons).indices
        self.assertEqual(len(set(train_idx) & set(eval_idx)), 0)
        self.assertEqual(sorted(train_idx + eval_idx), list(range(len(self.dataset))))
        for idx in eval_idx:
            self.assertIn(self.dataset.get_image_from_idx(idx).year(), EVAL_SEASONS)

    def test_random_split_by_image(self):
        train, test = self.dataset.random_split([0.8, 0.2], split_by='image',
                                                generator=torch.Generator().manual_seed(0))
        self.assertEqual(len(train) + len(test), len(self.dataset))
        self.assertEqual(len(set(train.indices) & set(test.indices)), 0)

    def test_random_split_by_sequence_has_no_leakage(self):
        train, test = self.dataset.random_split([0.5, 0.5], split_by='sequence',
                                                generator=torch.Generator().manual_seed(0))
        train_seqs = {self.dataset.get_image_from_idx(i).sequence_id() for i in train.indices}
        test_seqs = {self.dataset.get_image_from_idx(i).sequence_id() for i in test.indices}
        self.assertEqual(train_seqs & test_seqs, set())
        self.assertEqual(len(train) + len(test), len(self.dataset))

    def test_random_split_by_season_has_no_leakage(self):
        train, test = self.dataset.random_split([0.5, 0.5], split_by='season',
                                                generator=torch.Generator().manual_seed(0))
        train_seasons = {self.dataset.get_image_from_idx(i).year() for i in train.indices}
        test_seasons = {self.dataset.get_image_from_idx(i).year() for i in test.indices}
        self.assertEqual(train_seasons & test_seasons, set())
        self.assertEqual(len(train) + len(test), len(self.dataset))

    def test_ignore_list(self):
        ignored = [names[0] for names in self.files.values()]
        dataset = make_dataset(ignore_list=ignored)
        self.assertEqual(len(dataset), len(self.dataset) - len(ignored))
        kept = {os.path.basename(dataset.get_image_from_idx(i).filepath()) for i in range(len(dataset))}
        self.assertEqual(kept & set(ignored), set())

    def test_filter_func(self):
        dataset = make_dataset(filter_func=lambda image: image.grade() >= 3)
        expected = sum(1 for i in range(len(self.dataset)) if self.dataset.get_image_from_idx(i).grade() >= 3)
        self.assertEqual(len(dataset), expected)
        self.assertTrue(all(dataset.get_image_from_idx(i).grade() >= 3 for i in range(len(dataset))))

    def test_transform_func(self):
        dataset = make_dataset(transform_func=lambda image: np.clip(image, 170, 300) / 300)
        expected = np.clip(self.dataset[0][0], 170, 300) / 300
        np.testing.assert_allclose(dataset[0][0], expected)

    def test_load_data_into_memory(self):
        dataset = make_dataset(load_data_into_memory='all_data')
        self.assertEqual(len(dataset), len(self.dataset))
        for idx in (0, len(dataset) - 1):
            np.testing.assert_array_equal(dataset[idx][0], self.dataset[idx][0])

    def test_get_images_by_sequence(self):
        dataset = make_dataset(get_images_by_sequence=True)
        self.assertEqual(len(dataset), len(self.files))
        images, labels = dataset[0]
        seq = dataset.get_ith_sequence(0).get_sequence_str()
        self.assertEqual(images.shape, (len(self.files[seq]), 512, 512))
        self.assertEqual(labels.shape, (len(self.files[seq]),))

    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=4, shuffle=True)
        images, labels = next(iter(loader))
        self.assertEqual(tuple(images.shape), (4, 512, 512))
        self.assertEqual(tuple(labels.shape), (4,))

    def test_not_verbose_prints_nothing(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            dataset = make_dataset(get_images_by_sequence=True)
            dataset[0]
        self.assertEqual(out.getvalue(), '')


class TestFlaggedImages(TestCase):
    """Images without usable data, on a copy of the sample (WP, channel 1) where one image is replaced by a blank one."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.root = os.path.join(cls.tmp, 'WP')
        shutil.copytree(data_dir(1, 'WP'), cls.root)
        seq, names = next(iter(image_files(1, 'WP').items()))
        cls.blank_name = names[1]
        with h5py.File(os.path.join(cls.root, 'image', seq, cls.blank_name), 'w') as f:
            f.create_dataset('Data', data=np.full((512, 512), -0.4), compression='gzip')
        cls.args = (cls.root + '/image/', cls.root + '/metadata/', cls.root + '/metadata.json', 'pressure')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_sample_has_no_flagged_images(self):
        self.assertEqual(make_dataset().get_flagged_indices(), [])

    def test_flagged_images_are_kept_by_default(self):
        dataset = DigitalTyphoonDataset(*self.args)
        self.assertEqual(len(dataset), sum(len(f) for f in image_files(1, 'WP').values()))
        self.assertEqual([os.path.basename(p) for p in dataset.get_flagged_files()], [self.blank_name])
        indices = dataset.get_flagged_indices()
        self.assertEqual(len(indices), 1)
        self.assertEqual(os.path.basename(dataset.get_image_from_idx(indices[0]).filepath()), self.blank_name)

    def test_exclude_flagged(self):
        dataset = DigitalTyphoonDataset(*self.args, exclude_flagged=True)
        self.assertEqual(len(dataset), sum(len(f) for f in image_files(1, 'WP').values()) - 1)
        names = {os.path.basename(dataset.get_image_from_idx(i).filepath()) for i in range(len(dataset))}
        self.assertNotIn(self.blank_name, names)
        self.assertEqual(dataset.get_flagged_indices(), [])
