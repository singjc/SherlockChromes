import csv
import numpy as np
import os
import sqlite3

from evaluation_parser import overlaps

def parse_manual_annotation_file(manual_annotation_filename, rt_gap=3.4):
    filename_to_annotation = {}

    with open(manual_annotation_filename, 'r') as csv:
        next(csv)
        for line in csv:
            idx, filename, start, end, lib_rt, win_size = line.rstrip(
                '\r\n').split(',')

            win_size = int(win_size)
            lib_rt = float(lib_rt)
            lib_rt_idx = (win_size - 1) / 2

            if start and end:
                start, end = int(start), int(end)
                start_time = lib_rt - ((lib_rt_idx - start) * rt_gap)
                end_time = lib_rt - ((lib_rt_idx - end) * rt_gap)
            else:
                start_time, end_time = None, None

            filename_to_annotation[filename] = {
                'start_idx': start,
                'end_idx': end,
                'start_time': start_time,
                'end_time': end_time,
                'lib_rt_idx': lib_rt_idx,
                'lib_rt': lib_rt,
                'win_size': win_size
            }

    return filename_to_annotation

def get_feature_data_table(osw_filename, decoy=0):
    con = sqlite3.connect(osw_filename)
    cursor = con.cursor()

    query = \
        """SELECT r.FILENAME, p2.MODIFIED_SEQUENCE, p1.CHARGE, f.LEFT_WIDTH, 
        f.RIGHT_WIDTH, ms1.VAR_MASSDEV_SCORE, 
        ms1.VAR_ISOTOPE_CORRELATION_SCORE, ms1.VAR_ISOTOPE_OVERLAP_SCORE, 
        ms1.VAR_XCORR_COELUTION, ms1.VAR_XCORR_SHAPE,
        ms2.VAR_BSERIES_SCORE, ms2.VAR_DOTPROD_SCORE, ms2.VAR_INTENSITY_SCORE, 
        ms2.VAR_ISOTOPE_CORRELATION_SCORE, ms2.VAR_ISOTOPE_OVERLAP_SCORE, 
        ms2.VAR_LIBRARY_CORR, ms2.VAR_LIBRARY_DOTPROD, 
        ms2.VAR_LIBRARY_MANHATTAN, ms2.VAR_LIBRARY_RMSD, 
        ms2.VAR_LIBRARY_ROOTMEANSQUARE, ms2.VAR_LIBRARY_SANGLE, 
        ms2.VAR_LOG_SN_SCORE, ms2.VAR_MANHATTAN_SCORE, ms2.VAR_MASSDEV_SCORE, 
        ms2.VAR_MASSDEV_SCORE_WEIGHTED, ms2.VAR_NORM_RT_SCORE, 
        ms2.VAR_XCORR_COELUTION, ms2.VAR_XCORR_COELUTION_WEIGHTED, 
        ms2.VAR_XCORR_SHAPE, ms2.VAR_XCORR_SHAPE_WEIGHTED, ms2.VAR_YSERIES_SCORE
        FROM PRECURSOR p1
        LEFT JOIN PRECURSOR_PEPTIDE_MAPPING ppm ON p1.ID = ppm.PRECURSOR_ID
        LEFT JOIN PEPTIDE p2 ON p2.ID = ppm.PEPTIDE_ID
        LEFT JOIN FEATURE f ON p1.ID = f.PRECURSOR_ID
        LEFT JOIN RUN r on f.RUN_ID = r.ID
        LEFT JOIN FEATURE_MS1 ms1 ON f.ID = ms1.FEATURE_ID
        LEFT JOIN FEATURE_MS2 ms2 on f.ID = ms2.FEATURE_ID
        WHERE p1.DECOY = {0}""".format(decoy)
    res = cursor.execute(query)
    tmp = res.fetchall()
    con.close()
    
    return tmp

def create_feature_data(
    manual_annotation_dir,
    manual_annotation_filename,
    rt_gap,
    osw_dir,
    osw_filename,
    decoy,
    out_dir,
    data_npy,
    labels_npy,
    csv_filename):
    manual_annotation_filename = os.path.join(
        manual_annotation_dir, manual_annotation_filename)
    osw_filename = os.path.join(osw_dir, osw_filename)

    filename_to_annotation = parse_manual_annotation_file(
        manual_annotation_filename, rt_gap)

    feature_data_table = get_feature_data_table(osw_filename, decoy)

    num_samples = len(feature_data_table)

    feature_data_array = []

    labels = []

    feature_data_csv = []

    feature_idx = 0

    for i in range(num_samples):
        if None in feature_data_table[i]:
            continue
        
        filename, mod_seq, charge, left, right = feature_data_table[i][0:5]
        filename = '_'.join(
            [filename.split('/')[-1].split('.')[0], mod_seq, str(charge)])

        label = 0
        
        if filename not in filename_to_annotation:
            continue
        elif overlaps(
            left,
            right,
            filename_to_annotation[filename]['start_time'],
            filename_to_annotation[filename]['end_time']):
            label = 1

        left_idx = round(
            ((
                left - 
                filename_to_annotation[filename]['lib_rt']) / 
            rt_gap) + 
            filename_to_annotation[filename]['lib_rt_idx'])

        right_idx = round(
            ((
                right - 
                filename_to_annotation[filename]['lib_rt']) / 
            rt_gap) + 
            filename_to_annotation[filename]['lib_rt_idx'])

        labels.append([label])
        feature_data_csv.append(
            [
                feature_idx,
                filename,
                left_idx,
                right_idx,
                filename_to_annotation[filename]['start_idx'],
                filename_to_annotation[filename]['end_idx'],
                label,
                filename_to_annotation[filename]['lib_rt'],
                filename_to_annotation[filename]['win_size']
            ]
        )

        feature_idx+= 1

        feature_data_array.append(feature_data_table[i][5:])

    feature_data_array = np.array(feature_data_array)
    labels = np.array(labels)

    np.save(os.path.join(out_dir, data_npy), feature_data_array)
    np.save(os.path.join(out_dir, labels_npy), labels)

    with open(os.path.join(out_dir, csv_filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                'ID', 'Filename', 'OSW BB Start', 'OSW BB End',
                'Manual BB Start', 'Manual BB End', 'Label', 'Lib RT',
                'Window Size'
            ]
        )
        writer.writerows(feature_data_csv)
