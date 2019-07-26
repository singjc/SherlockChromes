import matplotlib.pyplot as plt
import sys
import time

sys.path.insert(0, '../datasets')
sys.path.insert(0, '../eda')

from analysis import create_histo
from visualizer import plot_binary_precision_recall_curve

def overlaps(
    pred_min,
    pred_max,
    target_min,
    target_max):
    if ((pred_min <= target_min and target_min <= pred_max <= target_max) or
        (pred_min <= target_min and pred_max >= target_max) or
        (target_min <= pred_min <= target_max and pred_max <= target_max) or
        (target_min <= pred_min <= target_max and pred_max >= target_max)):
        return True
    return False

def parse_model_evaluation_file(
    filenames,
    osw_threshold=2.5,
    mod_threshold=0.5,
    mod_min_pts=1,
    exclusion_list=None):
    exclude = {}
    if exclusion_list:
        with open(exclusion_list, 'r') as exclusions:
            next(exclusions)
            for line in exclusions:
                line = line.split(',')
                seq_source = line[1]
                seq = seq_source.split('_')[-2]
                exclude[seq] = True

    mod_stats = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    mod_tp, mod_fp, mod_tn, mod_fn = [], [], [], []

    mod_tp, mod_fp, mod_tn, mod_fn, osw_scores, target, pred = \
        [], [], [], [], [], [], []

    for filename in filenames:
        with open(filename, 'r') as infile:
            next(infile)
            for line in infile:
                line = line.rstrip('\r\n').split(',')
                
                (
                    chrom_id,
                    seq_source,
                    osw_start,
                    osw_end,
                    mod_start,
                    mod_end,
                    osw_score,
                    mod_score
                ) = line

                seq = seq_source.split('_')[-2]

                if seq not in exclude:
                    osw_start = int(osw_start)
                    osw_end = int(osw_end)

                    if mod_start:
                        mod_start = int(mod_start)
                    else:
                        mod_start = None

                    if mod_end:
                        mod_end = int(mod_end)
                    else:
                        mod_end = None

                    if mod_start and mod_end:
                        if (mod_end - mod_start + 1) < mod_min_pts:
                            mod_start, mod_end = None, None

                    osw_score = float(osw_score)
                    mod_score = float(mod_score)

                    osw_scores.append(osw_score)
                    pred.append(mod_score)

                    if osw_score < osw_threshold:
                        osw_start, osw_end = None, None

                    if mod_score < mod_threshold:
                        mod_start, mod_end = None, None

                    if not osw_start and mod_start:
                        mod_stats['fp']+= 1
                        mod_fp.append(
                            (
                                    chrom_id,
                                    osw_start,
                                    osw_end,
                                    mod_start,
                                    mod_end))
                        target.append(0)
                    elif osw_start and not mod_start:
                        mod_stats['fn']+= 1
                        mod_fn.append(
                            (
                                    chrom_id,
                                    osw_start,
                                    osw_end,
                                    mod_start,
                                    mod_end))
                        target.append(1)
                    elif not osw_start and not mod_start:
                        mod_stats['tn']+= 1
                        mod_tn.append(
                            (
                                    chrom_id,
                                    osw_start,
                                    osw_end,
                                    mod_start,
                                    mod_end))
                        target.append(0)
                    else:
                        if overlaps(osw_start, osw_end, mod_start, mod_end):
                            mod_stats['tp']+= 1
                            mod_tp.append(
                                (
                                    chrom_id,
                                    osw_start,
                                    osw_end,
                                    mod_start,
                                    mod_end))
                            target.append(1)
                        else:
                            mod_stats['fp']+= 1
                            mod_fp.append(
                                (
                                    chrom_id,
                                    osw_start,
                                    osw_end,
                                    mod_start,
                                    mod_end))
                            target.append(0)

    print(mod_stats)

    return mod_tp, mod_fp, mod_tn, mod_fn, osw_scores, target, pred

def parse_amended_model_evaluation_file(
    filenames,
    osw_threshold=2.1,
    mod_threshold=0.5,
    mod_min_pts=1,
    inclusion_idx_files=None,
    inclusion_list=None,
    plot_things=False):
    include_idx = {}
    if inclusion_idx_files:
        for inclusion_idx in inclusion_idx_files:
            with open(inclusion_idx, 'r') as inclusion_idxs:
                for line in inclusion_idxs:
                    line = line.rstrip('\r\n')
                    line = str(int(float(line)))
                    include_idx[line] = True

    include = {}
    if inclusion_list:
        with open(inclusion_list, 'r') as inclusions:
            next(inclusions)
            for line in inclusions:
                line = line.split(',')

                if line[0] in include_idx:
                    seq_source = line[1]
                    seq = seq_source.split('_')[-2]
                    include[seq] = True

    osw_stats = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    mod_stats = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    osw_tp, osw_fp, osw_tn, osw_fn = [], [], [], []
    mod_tp, mod_fp, mod_tn, mod_fn = [], [], [], []

    osw_target, mod_target, osw_pred, mod_pred = [], [], [], []

    for filename in filenames:
        with open(filename, 'r') as infile:
            next(infile)
            for line in infile:
                line = line.rstrip('\r\n').split(',')
                
                (
                    chrom_id,
                    seq_source,
                    osw_start,
                    osw_end,
                    mod_start,
                    mod_end,
                    osw_score,
                    mod_score,
                    manual_start,
                    manual_end,
                    manual_present
                ) = line

                if manual_present == '1':
                    seq = seq_source.split('_')[-2]

                    if seq in include:
                        osw_start = int(osw_start)
                        osw_end = int(osw_end)

                        if mod_start:
                            mod_start = int(mod_start)
                        else:
                            mod_start = None

                        if mod_end:
                            mod_end = int(mod_end)
                        else:
                            mod_end = None

                        if mod_start and mod_end:
                            if (mod_end - mod_start + 1) < mod_min_pts:
                                mod_start, mod_end = None, None

                        osw_score = float(osw_score)
                        mod_score = float(mod_score)

                        osw_pred.append(osw_score)
                        mod_pred.append(mod_score)

                        if osw_score <= osw_threshold:
                            osw_start, osw_end = None, None

                        if mod_score <= mod_threshold:
                            mod_start, mod_end = None, None

                        if manual_start and manual_end:
                            manual_start = int(manual_start)
                            manual_end = int(manual_end)
                        else:
                            manual_start, manual_end = None, None
                        
                        if manual_start == None and osw_start != None:
                            osw_stats['fp']+= 1
                            osw_fp.append(
                                (
                                    chrom_id,
                                    manual_start,
                                    manual_end,
                                    osw_start,
                                    osw_end))
                            osw_target.append(0)
                        elif manual_start != None and osw_start == None:
                            osw_stats['fn']+= 1
                            osw_fn.append(
                                (
                                    chrom_id,
                                    manual_start,
                                    manual_end,
                                    osw_start,
                                    osw_end))
                            osw_target.append(1)
                        elif manual_start == None and osw_start == None:
                            osw_stats['tn']+= 1
                            osw_tn.append(
                                (
                                    chrom_id,
                                    manual_start,
                                    manual_end,
                                    osw_start,
                                    osw_end))
                            osw_target.append(0)
                        else:
                            if overlap_more_than(
                                manual_start, manual_end, osw_start, osw_end):
                                osw_stats['tp']+= 1
                                osw_tp.append(
                                    (
                                    chrom_id,
                                    manual_start,
                                    manual_end,
                                    osw_start,
                                    osw_end))
                                osw_target.append(1)
                            else:
                                osw_stats['fp']+= 1
                                osw_fp.append(
                                    (
                                    chrom_id,
                                    manual_start,
                                    manual_end,
                                    osw_start,
                                    osw_end))
                                osw_target.append(0)

                        if manual_start == None and mod_start != None:
                            mod_stats['fp']+= 1
                            mod_fp.append(
                                (
                                    chrom_id,
                                    manual_start,
                                    manual_end,
                                    mod_start,
                                    mod_end))
                            mod_target.append(0)
                        elif manual_start != None and mod_start == None:
                            mod_stats['fn']+= 1
                            mod_fn.append(
                                (
                                    chrom_id,
                                    manual_start,
                                    manual_end,
                                    mod_start,
                                    mod_end))
                            mod_target.append(1)
                        elif manual_start == None and mod_start == None:
                            mod_stats['tn']+= 1
                            mod_tn.append(
                                (
                                    chrom_id,
                                    manual_start,
                                    manual_end,
                                    mod_start,
                                    mod_end))
                            mod_target.append(0)
                        else:
                            if overlap_more_than(
                                manual_start, manual_end, mod_start, mod_end):
                                mod_stats['tp']+= 1
                                mod_tp.append(
                                    (
                                    chrom_id,
                                    manual_start,
                                    manual_end,
                                    mod_start,
                                    mod_end))
                                mod_target.append(1)
                            else:
                                mod_stats['fp']+= 1
                                mod_fp.append(
                                    (
                                    chrom_id,
                                    manual_start,
                                    manual_end,
                                    mod_start,
                                    mod_end))
                                mod_target.append(0)

    print(osw_stats, mod_stats)

    if plot_things:
        plot_binary_precision_recall_curve(
            osw_target, osw_pred, mod_target, mod_pred)
        create_histo(osw_pred, title="OSW Scores")
        create_histo(mod_pred, title="Model Scores")

    return mod_tp, mod_fp, mod_tn, mod_fn, osw_tp, osw_fp, osw_tn, osw_fn

def decoys_per_target_metric(
    target_filename,
    decoy_filename):
    osw_targets, osw_decoys, mod_targets, mod_decoys = [], [], [], []

    with open(target_filename, 'r') as target_file:
        next(target_file)
        for line in target_file:
            line = line.rstrip('\r\n').split(',')
            osw_targets.append(float(line[-2]))
            mod_targets.append(float(line[-1]))

    with open(decoy_filename, 'r') as decoy_file:
        next(decoy_file)
        for line in decoy_file:
            line = line.rstrip('\r\n').split(',')
            osw_decoys.append(float(line[-2]))
            mod_decoys.append(float(line[-1]))
    
    osw_targets = np.array(osw_targets)
    osw_decoys = np.array(osw_decoys)
    mod_targets = np.array(mod_targets)
    mod_decoys = np.array(mod_decoys)

    num_osw_decoys_over_targets = []
    num_osw_targets = []
    num_mod_decoys_over_targets = []
    num_mod_targets = []

    for i in [n / 2 for n in range(12, -9, -1)]:
        num_osw_decoys = (osw_decoys >= i).sum()
        num_osw_targets.append((osw_targets >= i).sum())
        num_osw_decoys_over_targets.append(
            num_osw_decoys / (num_osw_decoys + num_osw_targets[-1]))

    for i in [0.05 * n for n in range(20, -1, -1)]:
        num_mod_decoys = (mod_decoys >= i).sum()
        num_mod_targets.append((mod_targets >= i).sum())
        num_mod_decoys_over_targets.append(
            num_mod_decoys / (num_mod_decoys + num_mod_targets[-1]))

    plt.plot(num_osw_targets, num_osw_decoys_over_targets, 'bo')
    plt.plot(num_mod_targets, num_mod_decoys_over_targets, 'r+')
    plt.show() 

if __name__ == '__main__':
    """
    Usage:

    tp, fp, tn, fn, osw_scores, target, pred = parse_model_evaluation_file(
        'evaluation_results/evaluation_results_all_32_decoy.csv',
        osw_threshold=2.1
        mod_threshold=0.5,
        mod_min_pts=5,
        exclusion_list='chromatograms_mixed.csv'
    )
    """
    start = time.time()

    print('It took {0:0.1f} seconds'.format(time.time() - start))

    pass
