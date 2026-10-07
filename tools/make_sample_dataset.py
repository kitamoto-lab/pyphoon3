"""
Build a small sample of the Digital Typhoon Dataset V3 with the same directory layout as the full dataset:

    <out>/<channel>/<basin>/image/<seq>/*.h5
    <out>/<channel>/<basin>/metadata/<seq>.csv
    <out>/<channel>/<basin>/metadata.json

A few short sequences per basin are kept (present in all four channels), with two sequences in the training seasons
and one in each SS-L2 evaluation season (2024, 2025). Observations are thinned to every `--every` hours.
By default only channel 1 is included; pass e.g. `--channels 1 2 3 4` to include more channels.
The h5 files are copied unchanged; track CSVs keep only the rows of the kept hours.

Usage:
    python make_sample_dataset.py --src /path/to/V3-2025 --out pyphoon3-sample-V3-2025
"""
import argparse
import csv
import json
import os
import shutil

SAMPLE_SEQUENCES = {
    'WP': ['202219', '202317', '202426', '202514'],
    'AU': ['202225', '202318', '202413', '202528'],
}


def copy_sequence(src_dir, out_dir, seq, channel, every):
    with open(f'{src_dir}/metadata/{seq}.csv', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [r for r in reader if int(r['hour']) % every == 0]
    os.makedirs(f'{out_dir}/metadata', exist_ok=True)
    with open(f'{out_dir}/metadata/{seq}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    os.makedirs(f'{out_dir}/image/{seq}', exist_ok=True)
    n_images = 0
    for r in rows:
        name = r[f'file_{channel}']  # the image column is named after the channel
        if name and os.path.exists(f'{src_dir}/image/{seq}/{name}'):
            shutil.copy2(f'{src_dir}/image/{seq}/{name}', f'{out_dir}/image/{seq}/{name}')
            n_images += 1
    return n_images


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True, help='full dataset root containing 1/ 2/ 3/ 4/')
    p.add_argument('--out', required=True)
    p.add_argument('--channels', type=int, nargs='+', default=[1], help='channels to include')
    p.add_argument('--every', type=int, default=6, help='keep observations every N hours')
    args = p.parse_args()

    for channel in args.channels:
        for basin, seqs in SAMPLE_SEQUENCES.items():
            src_dir = f'{args.src}/{channel}/{basin}'
            out_dir = f'{args.out}/{channel}/{basin}'
            with open(f'{src_dir}/metadata.json') as f:
                metadata = json.load(f)
            sample_metadata = {}
            for seq in seqs:
                entry = dict(metadata[seq])
                entry['images'] = copy_sequence(src_dir, out_dir, seq, channel, args.every)
                sample_metadata[seq] = entry
            with open(f'{out_dir}/metadata.json', 'w') as f:
                json.dump(sample_metadata, f, indent=1)
            print(f'{channel}/{basin}: ' + ', '.join(f'{s} ({e["images"]})' for s, e in sample_metadata.items()))


if __name__ == '__main__':
    main()
