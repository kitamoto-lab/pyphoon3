import json
import os
from datetime import datetime
from unittest import TestCase

from config_test import require_sample_data, data_dir, image_files
from pyphoon3.DigitalTyphoonSequence import DigitalTyphoonSequence


def setUpModule():
    require_sample_data()


class TestDigitalTyphoonSequence(TestCase):
    """DigitalTyphoonSequence on one sample sequence (WP, channel 1)."""

    def setUp(self):
        self.root = data_dir(1, 'WP')
        self.seq, self.names = next(iter(image_files(1, 'WP').items()))
        with open(f'{self.root}/metadata.json') as f:
            self.meta = json.load(f)[self.seq]

    def make_sequence(self, **kwargs):
        sequence = DigitalTyphoonSequence(self.seq, self.meta['season'], self.meta['images'], **kwargs)
        sequence.process_track_data(f'{self.root}/metadata/{self.seq}.csv')
        sequence.process_seq_img_dir_into_sequence(f'{self.root}/image/{self.seq}')
        return sequence

    def test_default_spectrum_reads_v3_images(self):
        sequence = self.make_sequence()
        self.assertEqual(sequence.get_image_at_idx(0).image().shape, (512, 512))

    def test_images_in_chronological_order(self):
        sequence = self.make_sequence()
        self.assertEqual(sequence.get_sequence_str(), self.seq)
        self.assertEqual(sequence.get_num_images(), len(self.names))
        self.assertTrue(sequence.num_images_match_num_expected())
        images = sequence.get_all_images_in_sequence()
        self.assertEqual([os.path.basename(i.filepath()) for i in images], self.names)
        self.assertEqual([i.datetime() for i in images],
                         [datetime.strptime(n[:10], '%Y%m%d%H') for n in self.names])

    def test_images_have_track_data(self):
        sequence = self.make_sequence()
        for image in sequence.get_all_images_in_sequence():
            self.assertEqual(image.year(), image.datetime().year)
            self.assertGreater(image.pressure(), 800)

    def test_multiple_label_types(self):
        sequence = self.make_sequence(label_type=('pressure', 'wind'))
        self.assertEqual(sequence.get_num_images(), len(self.names))

    def test_filter_func(self):
        sequence = DigitalTyphoonSequence(self.seq, self.meta['season'], self.meta['images'])
        sequence.process_track_data(f'{self.root}/metadata/{self.seq}.csv')
        sequence.process_seq_img_dir_into_sequence(f'{self.root}/image/{self.seq}',
                                                   filter_func=lambda image: image.hour() == 0)
        self.assertEqual(sequence.get_num_images(), sum(1 for n in self.names if n[8:10] == '00'))
