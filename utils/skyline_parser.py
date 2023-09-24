import bisect
import numpy as np
import os
import pandas as pd
import re
import click

from general_utils import get_subsequence_idxs
from pyopenms import AASequence

# annotations_dir="/media/justincsing/ExtraDrive1/Documents2/Roest_Lab/Github/PTMs_Project/Synth_PhosoPep/Justin_Synth_PhosPep/skyline_manual_annotations/Synthetic_Phospho_Dilution_Series/Manual_Annotation_Exports/filtered_exports"
# annotations_filename="chludwig_K150309_013_SW_0_filtered.tsv"

def split_file_name_info( filename ):
    repl_name = re.search(r'(chludwig_K\d+)_(\d+\w+?)_SW_\d*', filename, re.IGNORECASE)
    return repl_name.group(1), repl_name.group(2)

def parse_skyline_exported_annotations_JS(annotations_dir, annotations_filename):
    annotations = {}
    # Remove empty entries
    annotations_filename.remove("")
    for annotations_filename_i in annotations_filename:
        click.echo( "INFO: Processing %s annotation file." % (annotations_filename_i) )
        data = pd.read_csv( os.path.join(annotations_dir, annotations_filename_i), sep="\t" )
        repl, seq, charge, start, end = ( data['File Name'], data['Modified Sequence'], data['Precursor Charge'], data['Min Start Time'], data['Max End Time'] )
        seq = seq.apply( lambda SEQ: AASequence.fromString( SEQ ).toUniModString().decode('utf-8') )
        repl.replace({"[.]\w+$":""}, inplace=True, regex=True)
        repl_data = pd.DataFrame(repl.apply( lambda file: split_file_name_info(file) ).tolist())
        repl_data.columns=['repl_prefix', 'repl_num']
        repl_prefix = repl_data['repl_prefix']
        repl_num = repl_data['repl_num']
        key = repl_prefix.astype(str) + "_" + repl_num.astype(str) + "_" + seq.astype(str) + "_" + charge.astype(str)
        key = key.tolist()
        start = start.astype(float).tolist()
        end = end.astype(float).tolist()
        for key_i, start_i, end_i in zip(key, start, end):
            annotations[key_i] = ( {'start': start_i * 60, 'end': end_i * 60} )

    return annotations

def parse_skyline_exported_annotations(
    annotations_dir, annotations_filename, transition_results=False
):
    annotations = {}

    with open(os.path.join(annotations_dir, annotations_filename)) as infile:
        next(infile)

        for line in infile:
            line = line.rstrip('\r\n').split(',')

<<<<<<< HEAD
            repl, seq, charge, start, end = ( line[2], line[13], line[14], line[15], line[16] )
=======
            if transition_results:
                repl, rt, start, end, seq, charge = (
                    line[2], line[6], line[7], line[8], line[13], line[14])
            else:
                repl, rt, seq, charge, start, end = (
                    line[2], line[4], line[13], line[14], line[15], line[16])

>>>>>>> 70b30e2bf9b09488a036f6088af8ce3ab964f096
            repl = repl.replace('0R', '0PlasmaBiolR')
            seq = AASequence.fromString(seq).toUniModString().decode('utf-8')
            repl_prefix = (
                'hroest_K120808' if 'R04' not in repl else 'hroest_K120809')
            key = f'{repl_prefix}_{repl}_{seq}_{charge}'
<<<<<<< HEAD
            if start == '#N/A' or end == '#N/A':
                annotations[key] = {'start': None, 'end': None}
=======

            if start == end == rt == '#N/A':
                annotations[key] = {'start': None, 'end': None, 'rt': None}
>>>>>>> 70b30e2bf9b09488a036f6088af8ce3ab964f096
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
    osw_strong_labels_npy,
    osw_weak_labels_npy,
    out_dir,
    prefix='skyline_augmented',
    peak_only=False
):
    orig_strong_labels = np.load(os.path.join(osw_dir, osw_strong_labels_npy))
    orig_weak_labels = np.load(os.path.join(osw_dir, osw_weak_labels_npy))
    ms1_rt_arrays = {}
    decoy_counter = 0
    skyline_strong_counter = 0
    skyline_weak_counter = 0

    with open(os.path.join(osw_dir, osw_csv)) as infile:
        next(infile)

        for line in infile:
            line = line.rstrip('\r\n').split(',')
            idx, filename, lib_rt, win_size = (
                line[0], line[1], line[3], line[4])

            if 'DECOY' in filename:
                # By default, includes positive labels even for decoys
                # since based on OSW unscored boundaries
                orig_strong_labels[int(idx)] = np.zeros(
                    orig_strong_labels[int(idx)].shape)
                decoy_counter += 1
                continue
            elif filename not in annotations:
                continue
            elif not annotations[filename]['start']:
                orig_strong_labels[int(idx)] = np.zeros(
                    orig_strong_labels[int(idx)].shape)
                orig_weak_labels[int(idx)] = 0
                skyline_strong_counter += 1
                skyline_weak_counter += 1
                continue

            skyline_strong_counter += 1
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

            rt_segment = ms1_rt_arrays[repl][segment_l:segment_r]

            if (
                annotations[filename]['end'] < rt_segment[0]
                or annotations[filename]['start'] > rt_segment[-1]
                or (
                    peak_only
                    and not (
                        rt_segment[0]
                        <= annotations[filename]['rt']
                        <= rt_segment[-1]))
            ):
                orig_strong_labels[int(idx)] = np.zeros(
                    orig_strong_labels[int(idx)].shape)
                orig_weak_labels[int(idx)] = 0
                continue

            if peak_only:
                skyline_left_idx = skyline_right_idx = bisect.bisect(
                    rt_segment, annotations[filename]['rt'])
            else:
                skyline_left_idx = bisect.bisect_left(
                    rt_segment, annotations[filename]['start'])
                skyline_right_idx = bisect.bisect_left(
                    rt_segment, annotations[filename]['end'])

            if skyline_right_idx >= rt_segment.shape[0]:
                skyline_right_idx = rt_segment.shape[0] - 1

            orig_strong_labels[int(idx)] = np.where(
                np.logical_and(
                    rt_segment >= rt_segment[skyline_left_idx],
                    rt_segment <= rt_segment[skyline_right_idx]),
                1,
                0)

            new_weak_label = min(1, np.sum(orig_strong_labels[int(idx)]))

            if new_weak_label != orig_weak_labels[int(idx)]:
                orig_weak_labels[int(idx)] = new_weak_label
                skyline_weak_counter += 1

    print(
        f'Saving skyline augmented segmentation labels array of shape '
        f'{orig_strong_labels.shape} with '
        f'{skyline_strong_counter} Skyline substitutions '
        f'and {decoy_counter} decoy substitutions')

    np.save(
        os.path.join(out_dir, f'{prefix}_{osw_strong_labels_npy}'),
        orig_strong_labels.astype(np.int32))

    print(
        f'Saving skyline augmented classification labels array of shape '
        f'{orig_weak_labels.shape} with '
        f'{skyline_weak_counter} Skyline substitutions')

    np.save(
<<<<<<< HEAD
        os.path.join(out_dir, f'skyline_augmented_{osw_labels_npy}'),
        orig_labels.astype(np.int32)
    )

@click.command()
@click.option( '-annotations_dir', '--annotations_dir', required=True, help='Head directory containing manual annotation files.' )
@click.option( '-annotations_filename', '--annotations_filename', required=True, help='Comma separated filenames' )
@click.option( '-osw_dir', '--osw_dir', default="", help='' )
@click.option( '-osw_csv', '--osw_csv', default="binary_labels.npy", help='' )
@click.option( '-osw_labels_npy', '--osw_labels_npy', default="binary_labels.npy", help='' )
@click.option( '-out_dir', '--out_dir', default="binary_labels.npy", help='' )
def skyline_parser_main( annotations_dir, annotations_filename, osw_dir, osw_csv, osw_labels_npy, out_dir ):
    # Split annotation filenames string in separate filenames
    annotations_filename = annotations_filename.split(',')
    click.echo("INFO: Extracting skyline annotations...")
    annotations = parse_skyline_exported_annotations_JS( annotations_dir, annotations_filename )
    click.echo("INFO: Creating Skyline Augmented dataset...")
    create_skyline_augmented_osw_dataset(
    annotations,
    osw_dir,
    osw_csv,
    osw_labels_npy,
    out_dir
)

if __name__ == '__main__':
    skyline_parser_main()
=======
        os.path.join(out_dir, f'{prefix}_{osw_weak_labels_npy}'),
        orig_weak_labels.astype(np.int32))
>>>>>>> 70b30e2bf9b09488a036f6088af8ce3ab964f096
