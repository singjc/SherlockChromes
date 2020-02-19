import matplotlib.pyplot as plt
import sys
import time

sys.path.insert(0, '../datasets')
sys.path.insert(0, '../eda')

from analysis import create_histo
from general_utils import overlaps
from visualizer import plot_binary_precision_recall_curve

def get_filenames_from_idx(chromatograms_filename, idx_filenames=[]):
    idxs = {}
    for idx_filename in idx_filenames:
        with open(idx_filename, 'r') as idx_file:
            for line in idx_file:
                line = line.rstrip('\r\n')
                line = str(int(float(line)))
                idxs[line] = True

    filenames = {}
    with open(chromatograms_filename, 'r') as chromatograms_file:
        next(chromatograms_file)
        for line in chromatograms_file:
            line = line.split(',')
            
            if line[0] in idxs:
                filenames[line[1]] = True

    return filenames

def parse_model_evaluation_file(
    filenames,
    osw_threshold=2.1,
    mod_threshold=0.5,
    mod_min_pts=1,
    train_chromatogram_filenames=[],
    exclusion_idx_filenames=[]):
    excluded_filenames = {}

    for i in range(len(train_chromatogram_filenames)):
        excluded_filenames = {
            **excluded_filenames,
            **get_filenames_from_idx(
                train_chromatogram_filenames[i],
                exclusion_idx_filenames[i]
            )
        }

    mod_stats = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    mod_tp, mod_fp, mod_tn, mod_fn = [], [], [], []

    for filename in filenames:
        with open(filename, 'r') as infile:
            next(infile)
            for line in infile:
                line = line.rstrip('\r\n').split(',')
                
                (
                    chrom_id,
                    chromatogram_filename,
                    osw_start,
                    osw_end,
                    mod_start,
                    mod_end,
                    osw_score,
                    mod_score,
                    exp_rt,
                    win_size,
                    hq
                ) = line

                if chromatogram_filename in excluded_filenames:
                    continue

                osw_start = int(float(osw_start))
                osw_end = int(float(osw_end))

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
                elif osw_start and not mod_start:
                    mod_stats['fn']+= 1
                    mod_fn.append(
                        (
                                chrom_id,
                                osw_start,
                                osw_end,
                                mod_start,
                                mod_end))
                elif not osw_start and not mod_start:
                    mod_stats['tn']+= 1
                    mod_tn.append(
                        (
                                chrom_id,
                                osw_start,
                                osw_end,
                                mod_start,
                                mod_end))
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
                    else:
                        mod_stats['fp']+= 1
                        mod_fp.append(
                            (
                                chrom_id,
                                osw_start,
                                osw_end,
                                mod_start,
                                mod_end))

    print(mod_stats)

    return mod_tp, mod_fp, mod_tn, mod_fn

def parse_amended_model_evaluation_file(
    filenames=[],
    osw_threshold=2.1,
    mod_threshold=0.5,
    mod_min_pts=1,
    train_chromatogram_filename=None,
    inclusion_idx_filenames=[],
    plot_things=False):
    included_filenames = get_filenames_from_idx(
        train_chromatogram_filename, inclusion_idx_filenames)

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
                    chromatogram_filename,
                    osw_start,
                    osw_end,
                    mod_start,
                    mod_end,
                    osw_score,
                    mod_score,
                    manual_start,
                    manual_end,
                    exp_rt,
                    win_size,
                    manual_present
                ) = line

                if manual_present != '1':
                    continue

                if chromatogram_filename not in included_filenames:
                    continue

                osw_start = int(float(osw_start))
                osw_end = int(float(osw_end))

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
                    if overlaps(manual_start, manual_end, osw_start, osw_end):
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
                    if overlaps(manual_start, manual_end, mod_start, mod_end):
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
    decoy_filename,
    train_chromatogram_filename=None,
    exclusion_idx_filenames=[],
    logits=False):
    excluded_filenames = get_filenames_from_idx(
        train_chromatogram_filename, exclusion_idx_filenames)

    osw_targets, osw_decoys, mod_targets, mod_decoys = [], [], [], []

    with open(target_filename, 'r') as target_file:
        next(target_file)
        for line in target_file:
            line = line.rstrip('\r\n').split(',')

            if line[1] in excluded_filenames:
                continue

            osw_targets.append(float(line[6]))
            mod_targets.append(float(line[7]))

    with open(decoy_filename, 'r') as decoy_file:
        next(decoy_file)
        for line in decoy_file:
            line = line.rstrip('\r\n').split(',')

            if line[1] in excluded_filenames:
                continue
                
            osw_decoys.append(float(line[6]))
            mod_decoys.append(float(line[7]))
    
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
        num_osw_targets.append((osw_targets >= i).sum() + num_osw_decoys)
        num_osw_decoys_over_targets.append(
            num_osw_decoys / num_osw_targets[-1])

    if logits:
        for i in [n / 2 for n in range(6, -8, -1)]:
            num_mod_decoys = (mod_decoys >= i).sum()
            num_mod_targets.append((mod_targets >= i).sum() + num_mod_decoys)
            num_mod_decoys_over_targets.append(
                num_mod_decoys / num_mod_targets[-1])
    else:
        for i in [0.05 * n for n in range(20, -1, -1)]:
            num_mod_decoys = (mod_decoys >= i).sum()
            num_mod_targets.append((mod_targets >= i).sum() + num_mod_decoys)
            num_mod_decoys_over_targets.append(
                num_mod_decoys / num_mod_targets[-1])

    plt.plot(num_osw_targets, num_osw_decoys_over_targets, 'b-o', label='osw')
    plt.plot(num_mod_targets, num_mod_decoys_over_targets, 'r-+', label='mod')
    plt.xlabel('Number of Predicted Targets')
    plt.ylabel('Percentage Decoys in Targets')
    plt.xlim([-5000, max(num_mod_targets) + 25000])
    plt.ylim([-0.005, 0.35])
    plt.xticks([i for i in range(0, max(num_mod_targets) + 25000, 25000)])
    plt.yticks([i*0.01 for i in range(0, 36)])
    plt.grid()
    plt.legend(title='Inputs: ')
    plt.title('Pseudo 1 - Precision / Recall Curve')
    
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
