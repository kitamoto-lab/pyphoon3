import contextlib
import io
import os
import unittest
from unittest import TestCase

import numpy as np

from config_test import EVAL_SEASONS, require_sample_data, has_channels, dataset_args, multi_dataset_kwargs, image_files
from pyphoon3.DigitalTyphoonDataset import DigitalTyphoonDataset

CHANNELS = [1, 2, 3]


def setUpModule():
    require_sample_data()
    if not has_channels(CHANNELS):
        raise unittest.SkipTest(f'multi-channel tests need channels {CHANNELS} in the sample dataset '
                                f'(create it with tools/make_sample_dataset.py --channels 1 2 3 4)')


def make_multi_dataset(channels=CHANNELS, basin='WP', label='pressure', **kwargs):
    return DigitalTyphoonDataset('', '', '', label, **multi_dataset_kwargs(channels, basin), **kwargs)


def observation_times(channel, basin):
    """Set of (sequence ID, YYYYMMDDHH) with an image in the channel."""
    return {(seq, name[:10]) for seq, names in image_files(channel, basin).items() for name in names}


class TestDigitalTyphoonDatasetMultiChannel(TestCase):
    """Multi-channel dataset (channels 1, 2 and 3) on the sample data."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = make_multi_dataset()
        cls.single = {c: DigitalTyphoonDataset(*dataset_args(c, 'WP'), 'pressure') for c in CHANNELS}

    def test_len_is_intersection_of_channels(self):
        common = set.intersection(*[observation_times(c, 'WP') for c in CHANNELS])
        self.assertEqual(len(self.dataset), len(common))

    def test_getitem_stacks_channels(self):
        image, label = self.dataset[0]
        self.assertEqual(image.shape, (len(CHANNELS), 512, 512))
        self.assertEqual(label, self.dataset.get_image_from_idx(0).pressure())

    def test_channels_are_in_given_order(self):
        """Channel k of a multi-channel image equals the same observation in the k-th single-channel dataset."""
        for idx in (0, len(self.dataset) - 1):
            image, _ = self.dataset[idx]
            image_obj = self.dataset.get_image_from_idx(idx)
            seq, time = image_obj.sequence_id(), image_obj.datetime()
            for k, channel in enumerate(CHANNELS):
                single = [i for i in self.single[channel].image_objects_from_sequence(seq) if i.datetime() == time]
                self.assertEqual(len(single), 1)
                np.testing.assert_array_equal(image[k], single[0].image())

    def test_season_subsets_cover_dataset(self):
        eval_idx = self.dataset.images_from_seasons(EVAL_SEASONS).indices
        train_seasons = [s for s in self.dataset.get_seasons() if s not in EVAL_SEASONS]
        train_idx = self.dataset.images_from_seasons(train_seasons).indices
        self.assertEqual(sorted(set(train_idx) | set(eval_idx)), list(range(len(self.dataset))))
        self.assertEqual(set(train_idx) & set(eval_idx), set())

    def test_season_subsets_have_no_duplicates(self):
        # Every channel has its own metadata JSON listing the same sequences; each image must appear only once.
        eval_idx = self.dataset.images_from_seasons(EVAL_SEASONS).indices
        train_seasons = [s for s in self.dataset.get_seasons() if s not in EVAL_SEASONS]
        train_idx = self.dataset.images_from_seasons(train_seasons).indices
        self.assertEqual(len(eval_idx), len(set(eval_idx)))
        self.assertEqual(len(train_idx) + len(eval_idx), len(self.dataset))

    def test_sequences_registered_once(self):
        seqs = image_files(1, 'WP')
        self.assertEqual(self.dataset.get_number_of_sequences(), len(seqs))
        self.assertEqual(sorted(self.dataset.get_sequence_ids()), sorted(seqs))
        for season, ids in self.dataset.season_to_sequence_nums.items():
            self.assertEqual(len(ids), len(set(ids)))

    def test_random_split_by_season(self):
        train, test = self.dataset.random_split([0.5, 0.5], split_by='season')
        self.assertEqual(len(train) + len(test), len(self.dataset))
        train_seasons = {self.dataset.get_image_from_idx(i).year() for i in train.indices}
        test_seasons = {self.dataset.get_image_from_idx(i).year() for i in test.indices}
        self.assertEqual(train_seasons & test_seasons, set())

    def test_random_split_by_sequence_has_no_leakage(self):
        train, test = self.dataset.random_split([0.5, 0.5], split_by='sequence')
        train_seqs = {self.dataset.get_image_from_idx(i).sequence_id() for i in train.indices}
        test_seqs = {self.dataset.get_image_from_idx(i).sequence_id() for i in test.indices}
        self.assertEqual(train_seqs & test_seqs, set())

    @unittest.skipUnless(has_channels([1, 2, 3, 4], 'WP') and has_channels([1, 2, 3, 4], 'AU'),
                         'needs channels 1-4 in both basins')
    def test_all_four_channels_both_basins(self):
        for basin in ['WP', 'AU']:
            with self.subTest(basin=basin):
                dataset = make_multi_dataset([1, 2, 3, 4], basin)
                common = set.intersection(*[observation_times(c, basin) for c in [1, 2, 3, 4]])
                self.assertEqual(len(dataset), len(common))
                self.assertEqual(dataset[0][0].shape, (4, 512, 512))

    def test_not_verbose_prints_nothing(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            make_multi_dataset([1, 2])
        self.assertEqual(out.getvalue(), '')
