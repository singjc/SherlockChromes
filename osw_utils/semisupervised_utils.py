import numpy as np

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
            line = line.split(',')
            idx, filename, score = line[0], line[1], line[7]
            score = float(score)

            if idx in idxs and ('DECOY_' in filename or score >= 0.5):
                new_idxs.append(int(idx))

    np.savetxt(new_train_idx_filename, np.array(new_idxs))
