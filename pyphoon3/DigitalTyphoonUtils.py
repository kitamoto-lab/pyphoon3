import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from typing import List, Tuple

import h5py
import numpy as np

class SPLIT_UNIT(Enum):
    """
    Enum denoting which unit to treat as atomic when splitting the dataset
    """
    SEQUENCE = 'sequence'
    SEASON = 'season'
    IMAGE = 'image'

    @classmethod
    def has_value(cls, value):
        """
        Returns true if value is present in the enum

        :param value: str, the value to check for
        :return: bool
        """
        return value in cls._value2member_map_

class LOAD_DATA(Enum):
    """
    Enum denoting what level of data should be stored in memory
    """
    NO_DATA = False
    ONLY_TRACK = 'track'
    ONLY_IMG = 'images'
    ALL_DATA = 'all_data'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class TRACK_COLS(Enum):
    """
    Enum containing indices in a track csv col to find the respective data
    """
    YEAR = 0
    MONTH = 1
    DAY = 2
    HOUR = 3
    GRADE = 4
    LAT = 5
    LNG = 6
    PRESSURE = 7
    WIND = 8
    DIR50 = 9
    LONG50 = 10
    SHORT50 = 11
    DIR30 = 12
    LONG30 = 13
    SHORT30 = 14
    LANDFALL = 15
    INTERPOLATED = 16
    FILENAME = 17
    MASK_1 = 18
    MASK_1_PERCENT = 19

    @classmethod
    def str_to_value(cls, name):
        name_map = {
            'year': TRACK_COLS.YEAR.value,
            'month': TRACK_COLS.MONTH.value,
            'day': TRACK_COLS.DAY.value,
            'hour': TRACK_COLS.HOUR.value,
            'grade': TRACK_COLS.GRADE.value,
            'lat': TRACK_COLS.LAT.value,
            'lng': TRACK_COLS.LNG.value,
            'pressure': TRACK_COLS.PRESSURE.value,
            'wind': TRACK_COLS.WIND.value,
            'dir50': TRACK_COLS.DIR50.value,
            'long50': TRACK_COLS.LONG50.value,
            'short50': TRACK_COLS.SHORT50.value,
            'dir30': TRACK_COLS.DIR30.value,
            'long30': TRACK_COLS.LONG30.value,
            'short30': TRACK_COLS.SHORT30.value,
            'landfall': TRACK_COLS.LANDFALL.value,
            'interpolated': TRACK_COLS.INTERPOLATED.value,
            'filename': TRACK_COLS.FILENAME.value,
            'mask_1': TRACK_COLS.MASK_1.value,
            'mask_1_percent': TRACK_COLS.MASK_1_PERCENT.value,
        }
        if name in name_map:
            return name_map[name]
        else:
            raise KeyError(f"{name} is not a valid column name.")

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

def _verbose_print(string: str, verbose: bool):
    """
    Prints the string if verbose is true

    :param string: str
    :param verbose: bool
    :return: None
    """
    if verbose:
        print(string)

def parse_image_filename(filename: str, separator='-') -> Tuple[str, datetime, str]:
    """
    Takes the filename of a Digital Typhoon image and parses it to return the date it was taken, the sequence ID
    it belongs to, and the satellite that took the image

    :param filename: str, filename of the image
    :param separator: char, separator used in the filename
    :return: (str, datetime, str), Tuple containing the sequence ID, the datetime, and satellite string
    """
    try:
        date, sequence_num, satellite, _ = filename.split(separator)
        season = int(date[:4])
        date_month = int(date[4:6])
        date_day = int(date[6:8])
        date_hour = int(date[8:10])
        sequence_datetime = datetime(year=season, month=date_month,
                                     day=date_day, hour=date_hour)
        return sequence_num, sequence_datetime, satellite
    except ValueError:
        raise ValueError(
            f"Filename {filename} does not match the expected format.")

def parse_common_image_filename(filename: str, separator='-') -> Tuple[str, datetime, str]:
    """
    Takes the filename of a Digital Typhoon image and parses it to return the date it was taken, the sequence ID
    it belongs to, and the satellite that took the image

    :param filename: str, filename of the image
    :param separator: char, separator used in the filename
    :return: (str, datetime, str), Tuple containing the sequence ID, the datetime, and satellite string
    """
    try:
        date, sequence_num, satellite = filename.split(separator)
        season = int(date[:4])
        date_month = int(date[4:6])
        date_day = int(date[6:8])
        date_hour = int(date[8:10])
        sequence_datetime = datetime(year=season, month=date_month,
                                     day=date_day, hour=date_hour)
        return sequence_num, sequence_datetime, satellite
    except ValueError:
        raise ValueError(
            f"Filename {filename} does not match the expected format.")

def get_seq_str_from_track_filename(filename: str) -> str:
    """
    Given a track filename, returns the sequence ID it belongs to.

    :param filename: str, the filename (e.g., "sequence1.csv")
    :return: str, the sequence ID string (e.g., "sequence1")
    :raises ValueError: If the filename does not end with '.csv'
    """
    # Split the filename into root and extension
    sequence_num, ext = os.path.splitext(filename)

    # Validate the extension
    if ext.lower() != '.csv':
        raise ValueError(
            f"Unexpected file extension: '{ext}'. Expected a '.csv' file.")

    return sequence_num

def is_image_file(filename: str) -> bool:
    """
    Given a DigitalTyphoon file, returns if it is an h5 image.

    :param filename: str, the filename
    :return: bool, True if it is an h5 image, False otherwise
    """
    return filename.endswith(".h5")


# Images without usable data compress to small h5 files (about 10 KB, against about 400 KB for normal images),
# so only files below this size are opened when looking for flagged images.
FLAG_MAX_FILE_SIZE = 50_000


def is_flagged_image_array(image: np.ndarray) -> bool:
    """
    Returns whether an image has no usable data: blank (at most two distinct values) or mostly missing
    (more than half of the pixels at the image's minimum, i.e. fill, value).

    :param image: np.ndarray, image pixels
    :return: bool, True if the image should be flagged
    """
    return len(np.unique(image)) <= 2 or float((image == image.min()).mean()) > 0.5


def is_flagged_image_file(filepath: str, spectrum: str = 'Data') -> bool:
    """
    Returns whether an h5 image file has no usable data (see is_flagged_image_array). Files of at least
    FLAG_MAX_FILE_SIZE bytes are not opened and are never flagged.

    :param filepath: str, path to the h5 image file
    :param spectrum: str, name of the dataset in the h5 file
    :return: bool, True if the image should be flagged
    """
    if os.path.getsize(filepath) >= FLAG_MAX_FILE_SIZE:
        return False
    with h5py.File(filepath, 'r') as h5file:
        return is_flagged_image_array(h5file[spectrum][()])


def find_flagged_image_files(image_dirs: List[str], spectrum: str = 'Data', max_workers: int = 32) -> List[str]:
    """
    Returns the paths of the flagged images (see is_flagged_image_file) in image directories laid out as
    <image_dir>/<sequence ID>/<image>.h5. Files are checked in parallel threads, since the check is dominated by
    file system latency.

    :param image_dirs: list of image directories
    :param spectrum: str, name of the dataset in the h5 files
    :param max_workers: int, number of threads
    :return: sorted list of paths of the flagged image files
    """
    paths = []
    for image_dir in image_dirs:
        for seq in sorted(os.listdir(image_dir)):
            seq_dir = os.path.join(image_dir, seq)
            if os.path.isdir(seq_dir):
                paths.extend(os.path.join(seq_dir, f) for f in os.listdir(seq_dir) if is_image_file(f))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        flags = list(executor.map(lambda p: is_flagged_image_file(p, spectrum), paths))
    return sorted(p for p, flag in zip(paths, flags) if flag)
