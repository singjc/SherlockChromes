import argparse
import bisect
import csv
import numpy as np
import os
import sqlite3
import time

from general_utils import calc_bin_idx, get_subsequence_idxs

def get_specs_from_sql(con, cursor, repl):
    query = \
        f"""SELECT
        PRECURSOR_ID,
        MODIFIED_SEQUENCE,
        CHARGE,
        PRECURSOR_MZ,
        group_concat(PRODUCT_MZ, '|') AS PRODUCT_MZS,
        group_concat(LIBRARY_INTENSITY, '|') AS LIBRARY_INTENSITIES,
        DECOY,
		EXP_RT,
        DELTA_RT,
        LEFT_WIDTH,
        RIGHT_WIDTH
            FROM (
                SELECT 
                prec.ID AS PRECURSOR_ID,
                MODIFIED_SEQUENCE,
                prec.CHARGE AS CHARGE,
                PRECURSOR_MZ,
                PRODUCT_MZ,
                trans.LIBRARY_INTENSITY AS LIBRARY_INTENSITY,
                prec.DECOY AS DECOY,
                feat2.EXP_RT AS EXP_RT,
                feat2.DELTA_RT AS DELTA_RT,
                feat2.LEFT_WIDTH AS LEFT_WIDTH,
                feat2.RIGHT_WIDTH AS RIGHT_WIDTH
                FROM PRECURSOR AS prec
                LEFT JOIN PRECURSOR_PEPTIDE_MAPPING AS prec_to_pep
                ON prec.ID = prec_to_pep.PRECURSOR_ID
                LEFT JOIN PEPTIDE AS pep 
                ON prec_to_pep.PEPTIDE_ID = pep.ID
                LEFT JOIN TRANSITION_PRECURSOR_MAPPING AS trans_to_prec
                ON prec.ID = trans_to_prec.PRECURSOR_ID
                LEFT JOIN TRANSITION AS trans 
                ON trans_to_prec.TRANSITION_ID = trans.ID
                LEFT JOIN (
                    SELECT
                    PRECURSOR_ID,
                    EXP_RT,
                    DELTA_RT,
                    LEFT_WIDTH,
                    RIGHT_WIDTH
                    FROM FEATURE AS feat1
                    LEFT JOIN (
                        SELECT
                        ID
                        FROM RUN
                        WHERE FILENAME LIKE '%{repl}%') AS run
                    ON feat1.RUN_ID = run.ID
                    LEFT JOIN SCORE_MS2 AS score 
					ON feat1.ID = score.FEATURE_ID
                    WHERE NOT run.ID IS NULL AND score.RANK = 1
                    GROUP BY PRECURSOR_ID) AS feat2
                ON prec.ID = feat2.PRECURSOR_ID
                ORDER BY PRECURSOR_ID, TRANSITION_ID ASC)
        GROUP BY PRECURSOR_ID;"""
    res = cursor.execute(query)
    tmp = res.fetchall()

    return tmp

def extract_target_strip(
    lcms_map,
    target_mz,
    min_mz=0,
    bin_resolution=0.01,
    lower_span=55,
    upper_span=155):
    bin_idx = calc_bin_idx(target_mz, min_mz, bin_resolution)
    lower, upper = bin_idx - lower_span, bin_idx + upper_span
    strip = lcms_map[lower:upper]
    tgt_height = lower_span + upper_span

    if strip.shape[0] != tgt_height:
        width = strip.shape[-1]
        padded_strip = []

        if lower < 0:
            padded_strip.append(np.zeros((-lower, width)))
        
        padded_strip.append(strip)

        curr_height = sum([item.shape[0] for item in padded_strip])

        if curr_height < tgt_height:
            padded_strip.append(np.zeros((tgt_height - curr_height, width)))

        strip = np.concatenate(padded_strip, axis=0)

    return strip

def create_chromatogram(
    ms1_map,
    ms2_map,
    ms1_rt_array,
    prec_mz,
    lib_rt,
    prod_mzs,
    lib_intensities,
    charge,
    osw_label_left,
    osw_label_right,
    num_traces=6,
    min_swath_win=399.5,
    swath_win_size=25,
    analysis_win_size=175):
    ms2_map_idx = calc_bin_idx(prec_mz, min_swath_win, swath_win_size)

    chromatogram = []

    for mz in prod_mzs:
        chromatogram.append(extract_target_strip(ms2_map[ms2_map_idx], mz))

    if len(prod_mzs) < num_traces:
        for i in range(num_traces - len(prod_mzs)):
            chromatogram.append(np.zeros((chromatogram[-1].shape)))

    chromatogram.append(np.expand_dims(abs(ms1_rt_array - lib_rt), axis=0))

    if len(lib_intensities) < num_traces:
        for i in range(num_traces - len(prod_mzs)):
            lib_intensities.append(0)

    chromatogram.append(
        np.repeat(
            lib_intensities,
            ms1_rt_array.shape[-1]).reshape(len(lib_intensities), -1))

    # TODO: Also create channels for fragment charges
    chromatogram.append(
        np.repeat(charge, ms1_rt_array.shape[-1]).reshape(1, -1))

    chromatogram.append(extract_target_strip(ms1_map, prec_mz, min_mz=400))

    chromatogram = np.concatenate(chromatogram, axis=0)

    lib_rt_idx, osw_label_left_idx, osw_label_right_idx = None, None, None

    lib_rt_idx, ss_left_idx, ss_right_idx = get_subsequence_idxs(
        ms1_rt_array, lib_rt, analysis_win_size)

    if analysis_win_size >= 0:
        chromatogram = chromatogram[:, ss_left_idx:ss_right_idx]
        ms1_rt_array = ms1_rt_array[ss_left_idx:ss_right_idx]

    osw_label_left_idx = bisect.bisect_left(ms1_rt_array, osw_label_left)
    osw_label_right_idx = bisect.bisect_left(ms1_rt_array, osw_label_right)

    if osw_label_left_idx == osw_label_right_idx:
        osw_label_left_idx = osw_label_right_idx = None

    return chromatogram, lib_rt_idx, osw_label_left_idx, osw_label_right_idx

def create_repl_chromatograms_array(
    work_dir,
    osw_filename,
    repl,
    num_traces=6,
    min_swath_win=399.5,
    swath_win_size=25,
    analysis_win_size=175,
    create_label_arrays=False):
    repl = repl.split('.')[0]

    ms1_map = np.load(f'{repl}_ms1_array.npy')
    ms2_map = np.load(f'{repl}_ms2_array.npy')
    ms1_rt_array = np.load(f'{repl}_ms1_rt_array.npy')

    con = sqlite3.connect(os.path.join(work_dir, osw_filename))
    cursor = con.cursor()

    specs = get_specs_from_sql(con, cursor, repl)

    chromatograms_array = []
    out_csv = [
        [
            'ID',
            'Precursor ID',
            'Filename',
            'Library RT IDX',
            'Window Size',
            'Label Left IDX',
            'Label Right IDX'
        ]
    ]

    if create_label_arrays:
        segmentation_labels_array, classification_labels_array = [], []

    idx = 0
    for i in range(len(specs)):
        (
            prec_id,
            mod_seq,
            charge,
            prec_mz,
            prod_mzs,
            lib_intensities,
            decoy,
            exp_rt,
            delta_rt,
            left_width,
            right_width
        ) = specs[i]

        mod_seq_and_charge = f'{mod_seq}_{charge}'

        if exp_rt and delta_rt:
            lib_rt = exp_rt - delta_rt
        else:
            print(f'Precursor {prec_id}: {mod_seq_and_charge} missing lib_rt')
            continue

        prod_mzs = [float(x) for x in prod_mzs.split('|')]
        lib_intensities = [float(x) for x in lib_intensities.split('|')]

        chromatogram, lib_rt_idx, osw_label_left_idx, osw_label_right_idx = (
            create_chromatogram(
                ms1_map,
                ms2_map,
                ms1_rt_array,
                prec_mz,
                lib_rt,
                prod_mzs,
                lib_intensities,
                charge,
                left_width,
                right_width,
                num_traces,
                min_swath_win,
                swath_win_size,
                analysis_win_size
            )
        )

        filename = f'{repl}_{mod_seq_and_charge}'

        if decoy == 1:
            filename = f'DECOY_{filename}'

        chromatograms_array.append(chromatogram)

        if create_label_arrays:
            left = lib_rt_idx + (analysis_win_size // 2)
            right = lib_rt_idx + (analysis_win_size // 2) + 1
            ms1_rt_array_subsequence = ms1_rt_array[left:right]
            segmentation_labels_array.append(
                np.where(
                    np.logical_and(
                        ms1_rt_array >= ms1_rt_array[osw_label_left_idx],
                        ms1_rt_array <= ms1_rt_array[osw_label_right_idx]
                    ),
                    1,
                    0
                )
            )
            classification_labels_array.append(decoy)

        out_csv.append([
            idx,
            prec_id,
            filename,
            lib_rt_idx,
            analysis_win_size,
            osw_label_left_idx,
            osw_label_right_idx
        ])
        idx+= 1

    chromatograms_array = np.stack(chromatograms_array, axis=0)

    print(
        f'Saving chromatograms array for {repl} of shape '
        f'{chromatograms_array.shape}'
    )

    np.save(
        os.path.join(work_dir, f'{repl}_chromatograms_array'),
        chromatograms_array
    )

    if create_label_arrays:
        segmentation_labels_array = np.stack(segmentation_labels_array, axis=0)

        print(
            f'Saving segmentation labels array for {repl} of shape '
            f'{segmentation_labels_array.shape}'
        )

        np.save(
            os.path.join(work_dir, f'{repl}_segmentation_labels_array'),
            segmentation_labels_array
        )

        classificaton_labels_array = np.array(classification_labels_array)

        print(
            f'Saving classification labels array for {repl} of shape '
            f'{classificaton_labels_array.shape}'
        )

        np.save(
            os.path.join(work_dir, f'{repl}_classification_labels_array'),
            classification_labels_array
        )

    with open(os.path.join(work_dir, f'{repl}_chromatograms.csv'), 'w') as out:
        writer = csv.writer(out, lineterminator='\n')
        writer.writerows(out_csv)

def subset_chromatograms_and_create_labels():
    pass

if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('-work_dir', '--work_dir', type=str, default='.')
    parser.add_argument('-osw', '--osw', type=str, default='merged.osw')
    parser.add_argument('-repl_name', '--repl_name', type=str, default='')

    args = parser.parse_args()

    create_repl_chromatograms_array(args.work_dir, args.osw, args.repl_name)
    
    print('It took {0:0.1f} seconds'.format(time.time() - start))