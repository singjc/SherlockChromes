import argparse
import csv
import numpy as np
import os
import sqlite3
import time

from .general_utils import calc_bin_idx, get_subsequence_at

def get_specs_from_sql(con, cursor, repl):
    query = \
        f"""SELECT
        PRECURSOR_ID,
        MODIFIED_SEQUENCE || '_' || CHARGE AS PEPTIDE_NAME,
        PRECURSOR_MZ,
        group_concat(PRODUCT_MZ, '|') AS PRODUCT_MZS,
        group_concat(LIBRARY_INTENSITY, '|') AS LIBRARY_INTENSITIES,
        DECOY,
		EXP_RT,
        DELTA_RT
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
                feat2.DELTA_RT AS DELTA_RT
                FROM PRECURSOR AS prec
                LEFT JOIN PRECURSOR_PEPTIDE_MAPPING AS prec_to_pep
                ON prec.ID = prec_to_pep.PRECURSOR_ID
                LEFT JOIN PEPTIDE as pep 
                ON prec_to_pep.PEPTIDE_ID = pep.ID
                LEFT JOIN TRANSITION_PRECURSOR_MAPPING AS trans_to_prec
                ON prec.ID = trans_to_prec.PRECURSOR_ID
                LEFT JOIN TRANSITION as trans 
                ON trans_to_prec.TRANSITION_ID = trans.ID
                LEFT JOIN (
                    SELECT
                    PRECURSOR_ID,
                    EXP_RT,
                    DELTA_RT
                    FROM FEATURE as feat1
                    LEFT JOIN (
                        SELECT
                        ID
                        FROM RUN
                        WHERE FILENAME LIKE '%{repl}%') as run
                    ON feat1.RUN_ID = run.ID
                    WHERE NOT run.ID IS NULL
                    GROUP BY PRECURSOR_ID) as feat2
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
    half_span=5):
    bin_idx = calc_bin_idx(target_mz, min_mz, bin_resolution)
    strip = lcms_map[bin_idx - half_span:bin_idx + half_span + 1]

    return strip

def create_chromatogram(
    ms1_map,
    ms2_map,
    ms1_rt_array,
    prec_mz,
    lib_rt,
    prod_mzs,
    lib_intensities,
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
            ms1_rt_array.shape[-1])
        .reshape(
            len(lib_intensities),
            ms1_rt_array.shape[-1]))

    chromatogram.append(extract_target_strip(ms1_map, prec_mz, min_mz=400))

    chromatogram = np.concatenate(chromatogram, axis=0)

    if analysis_win_size >= 0:
        subsequence_left, subsequence_right, lib_rt_idx = get_subsequence_at(
            ms1_rt_array, lib_rt, analysis_win_size)

        chromatogram = chromatogram[:, subsequence_left:subsequence_right]

        return chromatogram, lib_rt_idx
    return chromatogram, None

def create_repl_chromatograms_array(
    work_dir,
    osw_filename,
    repl,
    num_traces=6,
    min_swath_win=399.5,
    swath_win_size=25,
    analysis_win_size=175):
    repl = repl.split('.')[0]

    ms1_map = np.load(f'{repl}_ms1_array.npy')
    ms2_map = np.load(f'{repl}_ms2_array.npy')
    ms1_rt_array = np.load(f'{repl}_ms1_rt_array.npy')

    con = sqlite3.connect(os.path.join(work_dir, osw_filename))
    cursor = con.cursor()

    specs = get_specs_from_sql(con, cursor, repl)

    chromatograms_array = []
    out_csv = [['ID', 'PREC_ID', 'Filename', 'Lib RT IDX', 'Window Size']]

    idx = 0
    for i in range(len(specs)):
        (
            prec_id,
            mod_seq_and_charge,
            prec_mz,
            prod_mzs,
            lib_intensities,
            decoy,
            exp_rt,
            delta_rt
        ) = specs[i]

        if exp_rt and delta_rt:
            lib_rt = exp_rt - delta_rt
        else:
            print(f'Precursor {prec_id}: {mod_seq_and_charge} missing lib_rt')
            continue

        prod_mzs = [float(x) for x in prod_mzs.split('|')]
        lib_intensities = [float(x) for x in lib_intensities.split('|')]

        chromatogram, lib_rt_idx = create_chromatogram(
            ms1_map,
            ms2_map,
            ms1_rt_array,
            prec_mz,
            lib_rt,
            prod_mzs,
            lib_intensities,
            num_traces,
            min_swath_win,
            swath_win_size,
            analysis_win_size
        )

        filename = f'{repl}_{mod_seq_and_charge}'

        if decoy == 1:
            filename = f'DECOY_{filename}'

        chromatograms_array.append(chromatogram)
        out_csv.append([idx, prec_id, filename, lib_rt_idx, analysis_win_size])
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