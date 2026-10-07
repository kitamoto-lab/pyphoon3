import os
import sys
from unittest import TestCase
from datetime import datetime

import numpy as np

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyphoon3.DigitalTyphoonUtils import *


class TestDigitalTyphoonUtils(TestCase):
    """Unit tests for the utility functions in DigitalTyphoonUtils."""

    def test_parse_image_filename(self):
        """Test parsing of a valid image filename into its components."""
        filename = '2008041301-200801-MTS1-1.h5'
        sequence_num, sequence_datetime, satellite = parse_image_filename(filename)

        # Assert the extracted components
        self.assertEqual(sequence_num, '200801')
        self.assertEqual(sequence_datetime, datetime(2008, 4, 13, 1))
        self.assertEqual(satellite, 'MTS1')

    def test_get_seq_str_from_track_filename(self):
        """Test extracting the sequence string from a track filename."""
        filename = '200801.csv'
        seq_str = get_seq_str_from_track_filename(filename)
        self.assertEqual(seq_str, '200801')

    def test_is_image_file(self):
        """Test whether a given filename is recognized as an image file."""
        # Test with a non-image file
        non_image_file = '200801.csv'
        self.assertFalse(is_image_file(non_image_file))

        # Test with a valid image file
        image_file = '2008041302-200801-MTS1-1.h5'
        self.assertTrue(is_image_file(image_file))

    def test_split_unit_has_value(self):
        """Test if SPLIT_UNIT contains a specific value."""
        # Check for existing and non-existing values
        self.assertTrue(SPLIT_UNIT.has_value('sequence'))
        self.assertFalse(SPLIT_UNIT.has_value('nonexistent_value'))

    def test_load_unit_has_value(self):
        """Test if LOAD_DATA contains a specific value."""
        # Check for existing and non-existing values
        self.assertTrue(LOAD_DATA.has_value('track'))
        self.assertFalse(LOAD_DATA.has_value('nonexistent_value'))

    def test_is_flagged_image_array(self):
        """Blank and mostly missing images are flagged; real images, even with few grey levels, are not."""
        rng = np.random.default_rng(0)
        self.assertTrue(is_flagged_image_array(np.full((512, 512), -0.4)))
        two_values = np.full((512, 512), 130.0); two_values[:3, :3] = -2.23
        self.assertTrue(is_flagged_image_array(two_values))
        mostly_fill = rng.uniform(190, 300, (512, 512)); mostly_fill[50:] = 130.0
        self.assertTrue(is_flagged_image_array(mostly_fill))
        self.assertFalse(is_flagged_image_array(rng.uniform(190, 300, (512, 512))))
        few_levels = np.round(rng.uniform(225, 247, (512, 512)))  # like the coarse water vapour images
        self.assertFalse(is_flagged_image_array(few_levels))
