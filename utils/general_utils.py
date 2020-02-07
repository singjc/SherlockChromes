import csv
import numpy as np
import os
import random
import sqlite3

from sklearn.model_selection import KFold

def overlaps(
    pred_min,
    pred_max,
    target_min,
    target_max,
    threshold=0.7):
    if not pred_min or not pred_max or not target_min or not target_max:
        return False

    overlap = min(pred_max, target_max) - max(pred_min, target_min)
    percent_overlap = overlap / (target_max - target_min)

    if percent_overlap >= threshold:
        return True
    return False

def get_high_quality_training_labels(
    target_csv,
    og_train_idx_filename,
    new_train_idx_filename):
    idxs = {}
    with open(og_train_idx_filename, 'r') as idx_file:
        for line in idx_file:
            line = line.rstrip('\r\n')
            line = str(int(float(line)))
            idxs[line] = True

    new_idxs = []
    with open(target_csv, 'r') as csv:
        next(csv)
        for line in csv:
            line = line.rstrip('\r\n').split(',')
            idx, filename, high_quality = line[0], line[1], line[10]
            high_quality = high_quality == '1'

            if idx in idxs and ('DECOY_' in filename or high_quality):
                new_idxs.append(int(idx))

    np.savetxt(new_train_idx_filename, np.array(new_idxs))

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

def get_naked_seq(seq):
    mod_start = seq.rfind('(')
    mod_end = seq.find(')')

    if mod_start != -1 and mod_end != -1:
        seq = seq[:mod_start] + seq[mod_end + 1:]

    return seq

def get_feature_data_table(osw_filename, decoy=0, spectral_info=True):
    con = sqlite3.connect(osw_filename)
    cursor = con.cursor()

    if spectral_info:
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
    else:
        query = \
        """SELECT r.FILENAME, p2.MODIFIED_SEQUENCE, p1.CHARGE, f.LEFT_WIDTH, 
        f.RIGHT_WIDTH, ms1.VAR_XCORR_COELUTION, ms1.VAR_XCORR_SHAPE,
        ms2.VAR_DOTPROD_SCORE, ms2.VAR_INTENSITY_SCORE, 
        ms2.VAR_LIBRARY_CORR, ms2.VAR_LIBRARY_DOTPROD, 
        ms2.VAR_LIBRARY_MANHATTAN, ms2.VAR_LIBRARY_RMSD, 
        ms2.VAR_LIBRARY_ROOTMEANSQUARE, ms2.VAR_LIBRARY_SANGLE, 
        ms2.VAR_LOG_SN_SCORE, ms2.VAR_MANHATTAN_SCORE, ms2.VAR_NORM_RT_SCORE, 
        ms2.VAR_XCORR_COELUTION, ms2.VAR_XCORR_COELUTION_WEIGHTED, 
        ms2.VAR_XCORR_SHAPE, ms2.VAR_XCORR_SHAPE_WEIGHTED
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
    threshold,
    out_dir,
    data_npy,
    labels_npy,
    csv_filename,
    spectral_info=True):
    manual_annotation_filename = os.path.join(
        manual_annotation_dir, manual_annotation_filename)
    osw_filename = os.path.join(osw_dir, osw_filename)

    filename_to_annotation = parse_manual_annotation_file(
        manual_annotation_filename, rt_gap)

    feature_data_table = get_feature_data_table(
        osw_filename, decoy, spectral_info)

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
            filename_to_annotation[filename]['end_time'],
            threshold):
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

def get_seqs_from_csv(seq_csv, naked=True, exclusion_seqs=None):
    excluded = {}
    if exclusion_seqs:
        for seq in exclusion_seqs:
            excluded[seq] = True

    seqs = {}
    with open(seq_csv, 'r') as seq_file:
        next(seq_file)
        for line in seq_file:
            seq = line.split(',')[1].split('_')[-2]

            if naked:
                seq = get_naked_seq(seq)

            if seq not in seqs and seq not in excluded:
                seqs[seq] = True

        seq_list = [seq for seq in seqs]

    return seq_list

def get_filenames_from_csv(seq_csv):
    filenames = {}
    with open(seq_csv, 'r') as seq_file:
        next(seq_file)
        for line in seq_file:
            filename = line.split(',')[1]

            if filename not in filenames:
                filenames[filename] = True

    return filenames

def get_train_val_test_sequence_splits(
    seq_csv,
    out_dir,
    naked=True,
    test_proportion=0.1,
    train_seqs_filename='train_seqs.txt',
    val_seqs_filename='val_seqs.txt',
    test_seqs_filename='test_seqs.txt'):
    seq_list = get_seqs_from_csv(seq_csv, naked)

    random.shuffle(seq_list)

    n = len(seq_list)
    n_test = int(n * test_proportion)
    n_train = n - 2 * n_test

    train_seqs = seq_list[:n_train]
    val_seqs = seq_list[n_train:(n_train + n_test)]
    test_seqs = seq_list[(n_train + n_test):]

    with open(os.path.join(out_dir, train_seqs_filename), 'w') as f:
        for seq in train_seqs:
            f.write(seq + '\n')

    with open(os.path.join(out_dir, val_seqs_filename), 'w') as f:
        for seq in val_seqs:
            f.write(seq + '\n')

    with open(os.path.join(out_dir, test_seqs_filename), 'w') as f:
        for seq in test_seqs:
            f.write(seq + '\n')

    train_seqs = {seq: True for seq in train_seqs}
    val_seqs = {seq: True for seq in val_seqs}
    test_seqs = {seq: True for seq in test_seqs}
    
    return train_seqs, val_seqs, test_seqs

def get_train_val_test_idx_from_sequences(
    split_csv,
    train_seqs,
    val_seqs,
    test_seqs,
    naked=True,
    out_dir='.',
    prefix='dl'):
    train_idx, val_idx, test_idx = [], [], []

    with open(split_csv, 'r') as seqs:
        next(seqs)
        for line in seqs:
            line = line.split(',')
            idx, seq = int(line[0]), line[1].split('_')[-2]

            if naked:
                seq = get_naked_seq(seq)

            if seq in train_seqs:
                train_idx.append(idx)
            elif seq in val_seqs:
                val_idx.append(idx)
            elif seq in test_seqs:
                test_idx.append(idx)

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    np.savetxt(
        os.path.join(out_dir, f'{prefix}_train_idx.txt'),
        np.array(train_idx),
        fmt='%i'
    )
    np.savetxt(
        os.path.join(out_dir, f'{prefix}_val_idx.txt'),
        np.array(val_idx),
        fmt='%i'
    )
    np.savetxt(
        os.path.join(out_dir, f'{prefix}_test_idx.txt'),
        np.array(test_idx),
        fmt='%i'
    )

    return train_idx, val_idx, test_idx

def create_train_val_test_split_by_sequence(
    in_dir,
    seq_csv,
    out_dir,
    split_csv=None,
    naked=True,
    test_proportion=0.1,
    prefix='dl'):
    seq_csv = os.path.join(in_dir, seq_csv)

    if split_csv:
        split_csv = os.path.join(in_dir, split_csv)

    train_seqs, val_seqs, test_seqs = get_train_val_test_sequence_splits(
        seq_csv, out_dir, naked, test_proportion)

    if split_csv:
        seq_csv = split_csv

    train_idx, val_idx, test_idx = get_train_val_test_idx_from_sequences(
        seq_csv, train_seqs, val_seqs, test_seqs, naked, out_dir, prefix)

def create_supervised_data_split(
    in_dir,
    dl_csv,
    trad_csv,
    out_dir,
    naked=True,
    test_proportion=0.1,
    dl_prefix='dl',
    trad_prefix='trad'):
    create_train_val_test_split_by_sequence(
        in_dir,
        dl_csv,
        out_dir,
        naked=naked,
        test_proportion=test_proportion,
        prefix=dl_prefix)

    create_train_val_test_split_by_sequence(
        in_dir,
        dl_csv,
        out_dir,
        split_csv=trad_csv,
        naked=naked,
        test_proportion=test_proportion,
        prefix=trad_prefix)

def get_kfold_sequence_splits(
    seq_csv,
    out_dir,
    exclusion_seqs=None,
    naked=True,
    n_splits=5,
    prefix='mv'):
    kf = KFold(n_splits=n_splits, shuffle=True)
    seq_splits = {}
    counter = 1

    seq_list = np.array(get_seqs_from_csv(seq_csv, naked, exclusion_seqs))

    for train_idx, test_idx in kf.split(seq_list):
        seq_splits[f'split_{counter}'] = {
            'train_seqs': {item: True for item in seq_list[train_idx]},
            'test_seqs': {item: True for item in seq_list[test_idx]}
        }
        counter+= 1

    for split in seq_splits:
        for split_part in seq_splits[split]:
            with open(
                os.path.join(
                    out_dir,
                    f'{prefix}_{split}_{split_part}.txt'
                ), 'w') as f:
                for seq in seq_splits[split][split_part]:
                    f.write(seq + '\n')

    return seq_splits

def get_kfold_idx_from_sequences(
    seq_csv,
    seq_splits,
    inclusion_filenames={},
    naked=True,
    out_dir='.',
    prefix='labeled'):
    idx_splits = {
        split: {'train_idx': [], 'test_idx': []} for split in seq_splits
    }

    with open(seq_csv, 'r') as seqs:
        next(seqs)
        for line in seqs:
            line = line.split(',')
            idx, filename = int(line[0]), line[1]
            seq = filename.split('_')[-2]

            if inclusion_filenames and filename not in inclusion_filenames:
                continue

            if naked:
                seq = get_naked_seq(seq)

            for split in seq_splits:
                for split_part in seq_splits[split]:
                    if seq in seq_splits[split][split_part]:
                        if 'train' in split_part:
                            idx_splits[split]['train_idx'].append(idx)
                        else:
                            idx_splits[split]['test_idx'].append(idx)

    for split in idx_splits:
        for split_part in idx_splits[split]:
            random.shuffle(idx_splits[split][split_part])
            np.savetxt(
                os.path.join(
                    out_dir, f'{prefix}_{split}_{split_part}.txt'
                ),
                np.array(idx_splits[split][split_part]),
                fmt='%i'
            )

    return idx_splits

def create_kfold_split_by_sequence(
    in_dir,
    seq_csv,
    out_dir,
    idx_csv=None,
    exclusion_seq_csvs=[],
    inclusion_filename_csvs=[],
    naked=True,
    n_splits=5,
    prefix='labeled'):
    seq_csv = os.path.join(in_dir, seq_csv)

    if idx_csv:
        idx_csv = os.path.join(in_dir, idx_csv)

    exclusion_seqs = []

    if exclusion_seq_csvs:
        for exclusion_seq_csv in exclusion_seq_csvs:
            exclusion_seq_csv = os.path.join(in_dir, exclusion_seq_csv)
            exclusion_seqs+= get_seqs_from_csv(exclusion_seq_csv, naked)

    seq_splits = get_kfold_sequence_splits(
        seq_csv,
        out_dir,
        exclusion_seqs,
        naked,
        n_splits,
        prefix)

    if idx_csv:
        seq_csv = idx_csv

    inclusion_filenames = {}
    if inclusion_filename_csvs:
        for inclusion_csv in inclusion_filename_csvs:
            inclusion_csv = os.path.join(in_dir, inclusion_csv)
            inclusion_filenames = {
                **inclusion_filenames,
                **get_filenames_from_csv(inclusion_csv)
            }

    idx_splits = get_kfold_idx_from_sequences(
        seq_csv, seq_splits, inclusion_filenames, naked, out_dir, prefix)
