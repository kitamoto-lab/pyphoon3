import os
from datetime import datetime
from unittest import TestCase

import h5py
import numpy as np

from config_test import require_sample_data, data_dir, image_files, track_rows
from pyphoon3.DigitalTyphoonImage import DigitalTyphoonImage


def setUpModule():
    require_sample_data()


class TestDigitalTyphoonImage(TestCase):
    """DigitalTyphoonImage on one sample image (WP, channel 1)."""

    def setUp(self):
        self.seq, names = next(iter(image_files(1, 'WP').items()))
        self.name = names[0]
        self.path = f'{data_dir(1, "WP")}/image/{self.seq}/{self.name}'
        self.row = track_rows(1, 'WP', self.seq)[self.name]
        self.track = np.array([float(v) if v not in ('',) and k not in ('file_1',) else 0.0
                               for k, v in self.row.items()][:17])

    def make_image(self, **kwargs):
        return DigitalTyphoonImage(self.path, track_data=self.track, sequence_id=self.seq, spectrum='Data', **kwargs)

    def test_image_matches_h5(self):
        with h5py.File(self.path, 'r') as f:
            expected = f['Data'][()].astype(np.float32)  # stored as float64, returned as float32
        image = self.make_image()
        self.assertEqual(image.image().dtype, np.float32)
        np.testing.assert_array_equal(image.image(), expected)
        self.assertEqual(image.image().shape, (512, 512))

    def test_load_into_memory(self):
        np.testing.assert_array_equal(self.make_image(load_imgs_into_mem=True).image(), self.make_image().image())

    def test_transform_func(self):
        image = self.make_image(transform_func=lambda a: a - 100)
        np.testing.assert_allclose(image.image(), self.make_image().image() - 100)

    def test_track_values(self):
        image = self.make_image()
        self.assertEqual(image.year(), int(self.row['year']))
        self.assertEqual(image.month(), int(self.row['month']))
        self.assertEqual(image.day(), int(self.row['day']))
        self.assertEqual(image.hour(), int(self.row['hour']))
        self.assertEqual(image.grade(), int(self.row['grade']))
        self.assertAlmostEqual(image.lat(), float(self.row['lat']))
        self.assertAlmostEqual(image.long(), float(self.row['lng']))
        self.assertEqual(image.pressure(), float(self.row['pressure']))
        self.assertEqual(image.wind(), float(self.row['wind']))
        self.assertEqual(image.datetime(), datetime.strptime(self.name[:10], '%Y%m%d%H'))

    def test_value_from_string(self):
        image = self.make_image()
        self.assertEqual(image.value_from_string('pressure'), image.pressure())
        self.assertEqual(image.value_from_string('wind'), image.wind())

    def test_metadata(self):
        image = self.make_image()
        self.assertEqual(image.sequence_id(), self.seq)
        self.assertEqual(os.path.basename(image.filepath()), self.name)
