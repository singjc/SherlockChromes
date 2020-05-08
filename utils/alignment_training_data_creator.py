import numpy as np 
import os

from utils.general_utils import get_filenames_from_csv, get_naked_seq

def parse_osw_score_data(
    in_dir,
    seq_csv,
    score_csv,
    train_seqs,
    test_seqs,
    out_dir,
    batch_size=15,
    num_files=7103,
    prefix='alignment_split_1',
    include_max=False,
    pad_with_decoys=False):
    filename_to_idx = {}
    seq_to_filename_to_osw_score = {}

    with open(os.path.join(in_dir, seq_csv), 'r') as seqs:
        next(seqs)
        for line in seqs:
            line = line.split(',')
            idx, filename = int(line[0]), line[1]
            seq = '_'.join(filename.split('_')[-2:])
            filename_to_idx[filename] = idx

            if seq in seq_to_filename_to_osw_score:
                seq_to_filename_to_osw_score[seq] = {
                    **seq_to_filename_to_osw_score[seq],
                    **{filename: -10}
                }
            else:
                seq_to_filename_to_osw_score[seq] = {filename: -10}
                
    with open(os.path.join(in_dir, score_csv), 'r') as scores:
        next(scores)
        for line in scores:
            line = line.split(',')
            filename, osw_score, label = line[1], float(line[6]), line[7]
            seq = '_'.join(filename.split('_')[-2:])

            if label == '0':
                continue
            if seq not in seq_to_filename_to_osw_score:
                continue
            if filename not in seq_to_filename_to_osw_score[seq]:
                continue

            seq_to_filename_to_osw_score[seq][filename] = osw_score

    train_seqs = np.loadtxt(os.path.join(in_dir, train_seqs), dtype=str)
    test_seqs = np.loadtxt(os.path.join(in_dir, test_seqs), dtype=str)
            
    train_idx, train_template_idx, val_idx, val_template_idx = [], [], [], []
    for seq in seq_to_filename_to_osw_score:
        max_scoring = max(
            seq_to_filename_to_osw_score[seq],
            key=lambda k: seq_to_filename_to_osw_score[seq][k]
        )
        total_files = len(seq_to_filename_to_osw_score[seq])

        if not include_max:
            total_files-= 1
        
        if get_naked_seq(seq.split('_')[0]) in train_seqs:
            train_template_idx.append(filename_to_idx[max_scoring])

            if include_max:
                train_idx+= [filename_to_idx[filename] 
                        for filename in seq_to_filename_to_osw_score[seq]]
            else:
                train_idx+= [filename_to_idx[filename]
                        for filename in seq_to_filename_to_osw_score[seq] 
                        if filename != max_scoring]

            if batch_size - total_files != 0 and pad_with_decoys:
                train_idx+= list(np.random.choice(
                    np.arange(num_files, num_files * 2),
                    (batch_size - total_files),
                    replace=False))
        elif get_naked_seq(seq.split('_')[0]) in test_seqs:
            val_template_idx.append(filename_to_idx[max_scoring])

            if include_max:
                val_idx+= [filename_to_idx[filename] 
                        for filename in seq_to_filename_to_osw_score[seq]]
            else:
                val_idx+= [filename_to_idx[filename]
                        for filename in seq_to_filename_to_osw_score[seq]
                        if filename != max_scoring]

            if batch_size - total_files != 0 and pad_with_decoys:
                val_idx+= list(np.random.choice(
                    np.arange(num_files * 2, num_files * 4),
                    (batch_size - total_files),
                    replace=False))

    np.savetxt(
        os.path.join(out_dir, f'{prefix}_train_idx.txt'),
        np.array(train_idx),
        fmt='%i'
    )
    np.savetxt(
        os.path.join(out_dir, f'{prefix}_train_template_idx.txt'),
        np.array(train_template_idx),
        fmt='%i'
    )
    np.savetxt(
        os.path.join(out_dir, f'{prefix}_val_idx.txt'),
        np.array(val_idx),
        fmt='%i'
    )
    np.savetxt(
        os.path.join(out_dir, f'{prefix}_val_template_idx.txt'),
        np.array(val_template_idx),
        fmt='%i'
    )
        