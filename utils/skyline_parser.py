import bisect
import numpy as np
import os

from general_utils import get_subsequence_idxs
from pyopenms import AASequence


def parse_skyline_exported_annotations(annotations_dir, annotations_filename):
    annotations = {}

    with open(os.path.join(annotations_dir, annotations_filename)) as infile:
        next(infile)

        for line in infile:
            line = line.rstrip('\r\n').split(',')

            repl, rt, seq, charge, start, end = (
                line[2], line[4], line[13], line[14], line[15], line[16])
            repl = repl.replace('0R', '0PlasmaBiolR')
            seq = AASequence.fromString(seq).toUniModString().decode('utf-8')
            repl_prefix = (
                'hroest_K120808' if 'R04' not in repl else 'hroest_K120809')
            key = f'{repl_prefix}_{repl}_{seq}_{charge}'

            if start == '#N/A' or end == '#N/A':
                annotations[key] = {'start': None, 'end': None, 'rt': None}
            else:
                annotations[key] = (
                    {
                        'start': float(start) * 60,
                        'end': float(end) * 60,
                        'rt': float(rt) * 60})

    return annotations


def create_skyline_augmented_osw_dataset(
    annotations,
    osw_dir,
    osw_csv,
    osw_labels_npy,
    out_dir,
    peak_only=False
):
    orig_labels = np.load(os.path.join(osw_dir, osw_labels_npy))
    ms1_rt_arrays = {}
    decoy_counter = 0
    skyline_counter = 0

    with open(os.path.join(osw_dir, osw_csv)) as infile:
        next(infile)

        for line in infile:
            line = line.rstrip('\r\n').split(',')
            idx, filename, lib_rt, win_size = (
                line[0], line[1], line[3], line[4])

            if 'DECOY' in filename:
                # By default, includes positive labels even for decoys
                # since based on OSW unscored boundaries
                orig_labels[int(idx)] = np.zeros(orig_labels[int(idx)].shape)
                decoy_counter += 1
                continue
            elif filename not in annotations:
                continue
            elif not annotations[filename]['start']:
                orig_labels[int(idx)] = np.zeros(orig_labels[int(idx)].shape)
                skyline_counter += 1
                continue

            skyline_counter += 1
            repl = '_'.join(filename.split('_')[:-2])

            if repl not in ms1_rt_arrays:
                ms1_rt_arrays[repl] = np.load(f'{repl}_ms1_rt_array.npy')

            try:
                lib_rt = int(lib_rt)
                lib_rt = ms1_rt_arrays[repl][lib_rt]
            except Exception:
                lib_rt = float(lib_rt)

            segment_l, segment_r = get_subsequence_idxs(
                ms1_rt_arrays[repl], lib_rt, subsequence_size=int(win_size))

            rt_segment = ms1_rt_arrays[repl][segment_l:segment_r + 1]

            if (
                annotations[filename]['end'] < rt_segment[0]
                or annotations[filename]['start'] >= rt_segment[-1]
                or (
                    peak_only
                    and not (
                        rt_segment[0]
                        <= annotations[filename]['rt']
                        < rt_segment[-1]))
            ):
                orig_labels[int(idx)] = np.zeros(orig_labels[int(idx)].shape)
                skyline_counter += 1
                continue

            rt_segment = rt_segment[:-1]

            if peak_only:
                skyline_left_idx = skyline_right_idx = bisect.bisect(
                    rt_segment, annotations[filename]['rt'])
            else:
                skyline_left_idx = bisect.bisect_left(
                    rt_segment, annotations[filename]['start'])
                skyline_right_idx = bisect.bisect_left(
                    rt_segment, annotations[filename]['end'])

            if skyline_left_idx >= rt_segment.shape[0]:
                skyline_left_idx = rt_segment.shape[0] - 1

            if skyline_right_idx >= rt_segment.shape[0]:
                skyline_right_idx = rt_segment.shape[0] - 1

            orig_labels[int(idx)] = np.where(
                np.logical_and(
                    rt_segment >= rt_segment[skyline_left_idx],
                    rt_segment <= rt_segment[skyline_right_idx]
                ),
                1,
                0
            )

    print(
        f'Saving skyline augmented segmentation labels array of shape '
        f'{orig_labels.shape} with {skyline_counter} Skyline substitutions '
        f'and {decoy_counter} decoy substitutions')

    np.save(
        os.path.join(out_dir, f'skyline_augmented_{osw_labels_npy}'),
        orig_labels.astype(np.int32))
