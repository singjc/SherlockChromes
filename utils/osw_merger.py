import csv
import numpy as np
import os

from pyopenms import AASequence

def merge_chromatogram_and_decoy_chromatogram_file(
    manually_validated_file,
    manually_validated_labels_npy,
    chromatogram_file_dir,
    chromatogram_file,
    decoy_chromatogram_file_dir,
    decoy_chromatogram_file,
    chromatogram_npy,
    osw_threshold,
    new_prefix):
    manual_files_to_info = {}

    with open(manually_validated_file, 'r') as manually_validated:
        next(manually_validated)
        for line in manually_validated:
            idx, filename, _, _  = line.rstrip('\r\n').split(',')
            manual_files_to_info[filename] = {
                'id': int(idx),
                'start': bbox_start,
                'end': bbox_end
            }

    merged_file = []

    counter, num_decoys = 0, 0

    manual_labels = np.load(manually_validated_labels_npy)
    chromatogram_labels = np.load(
        os.path.join(chromatogram_file_dir, chromatogram_npy)
    )

    with open(
        os.path.join(chromatogram_file_dir, chromatogram_file),
        'r') as chromatograms:
        next(chromatograms)
        for line in chromatograms:
            idx, filename, start, end, score, exp_rt, win_size = line.rstrip(
                '\r\n').split(',')
            idx = int(idx)
            score = float(score)
            manual = 0
                
            if filename in manual_files_to_info:
                chromatogram_labels[idx, :] = (
                    manual_labels[manual_files_to_info[filename]['id'], :])
                start, end = (
                    manual_files_to_info[filename]['start'],
                    manual_files_to_info[filename]['end']
                )
                manual = 1
            elif score < osw_threshold:
                chromatogram_labels[idx, :] = np.zeros(
                    (1, chromatogram_labels.shape[1])
                )

            merged_file.append(
                [idx, filename, start, end, score, exp_rt, win_size, manual]
            )
            counter+= 1

    with open(
        os.path.join(decoy_chromatogram_file_dir, decoy_chromatogram_file),
        'r') as decoy_chromatograms:
        next(decoy_chromatograms)
        for line in decoy_chromatograms:
            line = line.rstrip('\r\n').split(',')
            merged_file.append([counter] + line[1:] + [0])
            counter+= 1
            num_decoys+= 1

    with open(new_prefix + chromatogram_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
                'ID',
                'Filename',
                'BB Start',
                'BB End',
                'OSW Score',
                'Lib RT',
                'Window Size',
                'Manual'
            ])
        writer.writerows(merged_file)

    decoy_labels = np.zeros((num_decoys, chromatogram_labels.shape[1]))
    final_labels = np.vstack((chromatogram_labels, decoy_labels))

    assert final_labels.shape == (
        chromatogram_labels.shape[0] + num_decoys,
        chromatogram_labels.shape[1])

    np.save(new_prefix + chromatogram_npy, final_labels.astype(np.int32))

if __name__ == '__main__':
    merge_chromatogram_and_decoy_chromatogram_file(
        'manually_validated.csv',
        'manually_validated.npy',
        '',
        'chromatograms.csv',
        '',
        'chromatograms_decoy.csv',
        'osw_labels.npy',
        2.1,
        'merged_'
    )
