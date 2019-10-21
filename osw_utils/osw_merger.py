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
    manual_files_to_idx = {}

    with open(manually_validated_file, 'r') as manually_validated:
        next(manually_validated)
        for line in manually_validated:
            idx, filename, _, _  = line.rstrip('\r\n').split(',')

            filename = filename.split('_')
            mod_seq = AASequence.fromString(
                filename[-2]).toBracketString().decode('utf-8') 
            charge = filename[-1]
            filename = '_'.join(filename[:-2]).replace('0R', '0PlasmaBiolR')

            osw_filename = '_'.join([filename, mod_seq, charge])

            manual_files_to_idx[osw_filename] = int(idx)

    merged_file = []

    counter, num_decoys = 0, 0

    manual_labels = np.load(manually_validated_labels_npy)
    chromatogram_labels = np.load(os.path.join(chromatogram_file_dir, chromatogram_npy))

    with open(
        os.path.join(chromatogram_file_dir, chromatogram_file),
        'r') as chromatograms:
        next(chromatograms)
        for line in chromatograms:
            idx, filename, start, end, score, exp_rt = line.rstrip(
                '\r\n').split(',')
            idx = int(idx)
            score = float(score)
            manual = 0
                
            if filename in manual_files_to_idx:
                chromatogram_labels[idx, :] = \
                    manual_labels[manual_files_to_idx[filename], :]
                manual = 1
            elif score < osw_threshold:
                chromatogram_labels[idx, :] = np.zeros(
                    (1, chromatogram_labels.shape[1]))

            filename = '/'.join([chromatogram_file_dir, filename])
            merged_file.append(
                [idx, filename, start, end, score, manual, exp_rt])
            counter+= 1

    with open(
        os.path.join(decoy_chromatogram_file_dir, decoy_chromatogram_file),
        'r') as decoy_chromatograms:
        next(decoy_chromatograms)
        for line in decoy_chromatograms:
            line = line.rstrip('\r\n').split(',')
            line[1] = '/'.join([decoy_chromatogram_file_dir, line[1]])
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
                'Manual'
            ])
        writer.writerows(merged_file)

    decoy_labels = np.zeros((num_decoys, chromatogram_labels.shape[1]))
    final_labels = np.vstack((chromatogram_labels, decoy_labels))

    assert final_labels.shape == (
        chromatogram_labels.shape[0] + num_decoys,
        chromatogram_labels.shape[1])

    np.save(new_prefix + chromatogram_npy.split('.')[0], final_labels)

if __name__ == '__main__':
    merge_chromatogram_and_decoy_chromatogram_file(
        'manually_validated.csv',
        'manually_validated.npy',
        'OpenSWATHAutoAnnotatedAllXGB',
        'chromatograms.csv',
        'OpenSWATHAutoAnnotatedAllXGBDecoy',
        'chromatograms_decoy.csv',
        'osw_point_labels.npy',
        2.1,
        'merged_'
    )
