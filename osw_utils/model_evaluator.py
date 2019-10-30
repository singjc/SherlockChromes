import argparse
import numpy as np
import os
import pandas as pd
import scipy.ndimage
import sys
import time
import torch

from torch.utils.data import DataLoader

sys.path.insert(0, '../datasets')
sys.path.insert(0, '../models')
sys.path.insert(0, '../train')

from chromatograms_dataset import ChromatogramsDataset
from collate_fns import PadChromatogramsFor1DCNN
from transforms import ToTensor

def create_output_array(
    dataset,
    model,
    batch_size=32,
    device='cpu',
    load_npy=False,
    npy_dir='.',
    npy_name='output_array',
    mode='inference',
    threshold=0.5):
    output_array = None

    if load_npy:
        output_array = np.load(os.path.join(npy_dir, npy_name + '.npy'))

        return output_array
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PadChromatogramsFor1DCNN())

    for batch in dataloader:
        chromatograms = torch.from_numpy(
            np.asarray(batch[0])).float().to(device)

        output = model(chromatograms)

        if type(output_array) == type(None):
            output_array = output.detach().to('cpu').numpy()
        else:
            output_array = np.vstack(
                (output_array, output.detach().to('cpu').numpy()))

    if mode == 'semisupervised':
        output_array_labels = np.zeros((output_array.shape))

        binarized_output_array = np.where(output_array >= threshold, 1, 0)
        
        for i in range(len(dataset)):
            if 'Decoy' in dataset.chromatograms.iloc[i, 1]:
                break
            
            output = output_array[i]
            binarized_output = binarized_output_array[i]

            regions_of_interest = scipy.ndimage.find_objects(
                scipy.ndimage.label(binarized_output)[0])

            if not regions_of_interest:
                continue

            scores = [np.sum(output[r]) for r in regions_of_interest]
            best_region_idx = np.argmax(scores)
            best_region = regions_of_interest[best_region_idx][0]

            start_idx, end_idx = best_region.start, best_region.stop

            if end_idx - start_idx < 2 or end_idx - start_idx >= 60:
                continue

            end_idx+= 1

            output_array_labels[i, start_idx:end_idx] = np.ones(
                (1, end_idx - start_idx))

        output_array = output_array_labels

    np.save(os.path.join(npy_dir, npy_name), output_array)

    return output_array

def create_rpn_results_file(
    dataset,
    model,
    batch_size=32,
    device='cpu',
    data_dir='OpenSWATHAutoAnnotated',
    chromatograms_csv='chromatograms.csv',
    out_dir='.',
    results_csv='evaluation_results_rpn.csv'):
    chromatograms = pd.read_csv(os.path.join(
        data_dir, chromatograms_csv))

    assert len(chromatograms) == len(dataset)

    model_bounding_boxes = \
        [
            [
                'ID',
                'Filename',
                'Label BBox Start',
                'Label BBox End',
                'Pred BBox Start',
                'Pred BBox End',
                'OSW Score',
                'Model Score',
                'Lib RT'
            ]
        ]

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    idx = 0
    for batch in dataloader:
        chromatogram = torch.from_numpy(
            np.asarray(batch[0])).float().to(device)

        output = model(chromatogram)

        output_idx = 0
        for i in range(idx, idx + len(batch[1])):
            print(idx)

            row = chromatograms.iloc[i]

            left_width, right_width, score = output[output_idx, 0, 1:]

            model_bounding_boxes.append([
                    row['ID'],
                    row['Filename'],
                    row['BB Start'],
                    row['BB End'],
                    int(round(left_width)),
                    int(round(right_width)),
                    row['OSW Score'],
                    score,
                    row['Lib RT']])
            
            output_idx+= 1
            idx+= 1

    model_bounding_boxes = pd.DataFrame(model_bounding_boxes)

    model_bounding_boxes.to_csv(
        os.path.join(out_dir, results_csv),
        index=False,
        header=False)

def create_results_file(
    output_array,
    threshold=0.5,
    data_dir='OpenSWATHAutoAnnotated',
    chromatograms_csv='chromatograms.csv',
    out_dir='.',
    results_csv='evaluation_results.csv'):
    chromatograms = pd.read_csv(os.path.join(
        data_dir, chromatograms_csv))

    assert len(chromatograms) == output_array.shape[0]

    model_bounding_boxes = \
        [
            [
                'ID',
                'Filename',
                'Label BBox Start',
                'Label BBox End',
                'Pred BBox Start',
                'Pred BBox End',
                'OSW Score',
                'Model Score',
                'Lib RT'
            ]
        ]

    binarized_output_array = np.where(output_array >= threshold, 1, 0)

    for i in range(len(chromatograms)):
        print(i)

        row = chromatograms.iloc[i]

        output = output_array[i]
        binarized_output = binarized_output_array[i]

        left_width, right_width = None, None

        score = 0.0

        regions_of_interest = scipy.ndimage.find_objects(
            scipy.ndimage.label(binarized_output)[0])

        if regions_of_interest:
            scores = [np.sum(output[r]) for r in regions_of_interest]
            best_region_idx = np.argmax(scores)
            best_region = regions_of_interest[best_region_idx][0]

            start_idx, end_idx = best_region.start, best_region.stop

            if end_idx - start_idx >= 2 or end_idx - start_idx < 60:
                left_width, right_width = start_idx, end_idx

                score = np.divide(
                    np.sum(output[best_region]), output[best_region].shape[0])

        model_bounding_boxes.append([
                row['ID'],
                row['Filename'],
                row['BB Start'],
                row['BB End'],
                left_width,
                right_width,
                row['OSW Score'],
                str(score),
                row['Lib RT']])

    model_bounding_boxes = pd.DataFrame(model_bounding_boxes)

    model_bounding_boxes.to_csv(
        os.path.join(out_dir, results_csv),
        index=False,
        header=False)

def create_stats_eval_file(
    output_array,
    num_points=3,
    data_dir='OpenSWATHAutoAnnotated',
    chromatograms_csv='chromatograms.csv',
    out_dir='.',
    results_csv='evaluation_results.csv'):
    chromatograms = pd.read_csv(os.path.join(
        data_dir, chromatograms_csv))

    assert len(chromatograms) == output_array.shape[0]

    model_bounding_boxes = \
        [
            [
                'ID',
                'Filename',
                'Label BBox Start',
                'Label BBox End',
                'Pred BBox Start',
                'Pred BBox End',
                'OSW Score',
                'Model Score',
                'Lib RT'
            ]
        ]

    for i in range(len(chromatograms)):
        print(i)

        row = chromatograms.iloc[i]

        output = output_array[i, :]

        largest_idx = np.argmax(output)

        left_width = largest_idx - (num_points // 2)
        right_width = largest_idx + (num_points // 2)

        model_bounding_boxes.append([
                row['ID'],
                row['Filename'],
                row['BB Start'],
                row['BB End'],
                left_width,
                right_width,
                row['OSW Score'],
                str(output[largest_idx]),
                row['Lib RT']])

    model_bounding_boxes = pd.DataFrame(model_bounding_boxes)

    model_bounding_boxes.to_csv(
        os.path.join(out_dir, results_csv),
        index=False,
        header=False)

def create_semisupervised_results_file(
    output_array,
    data_dir='OpenSWATHAutoAnnotated',
    chromatograms_csv='chromatograms.csv',
    out_dir='.',
    results_csv='semisupervised_chromatograms.csv'):

    chromatograms = pd.read_csv(os.path.join(
        data_dir, chromatograms_csv))

    assert len(chromatograms) == output_array.shape[0]

    model_bounding_boxes = \
        [
            [
                'ID',
                'Filename',
                'Label BBox Start',
                'Label BBox End',
                'Pred BBox Start',
                'Pred BBox End',
                'OSW Score',
                'Model Score',
                'Lib RT'
            ]
        ]

    for i in range(len(chromatograms)):
        print(i)

        row = chromatograms.iloc[i]

        output = output_array[i]

        if max(output) < threshold and 'Decoy' not in row['Filename']:
            continue

        largest_idx = np.argmax(output)

        if output[largest_idx] == 1:
            min_idx, max_idx = 0, len(output) - 2

            while start_idx > min_idx and output[start_idx - 1] == 1:
                start_idx-= 1
            
            while end_idx < max_idx and output[end_idx + 1] == 1:
                end_idx+= 1

            left_width = start_idx
            right_width = end_idx
        else:
            left_width, right_width = None, None

        model_bounding_boxes.append([
                row['ID'],
                row['Filename'],
                row['BB Start'],
                row['BB End'],
                left_width,
                right_width,
                row['OSW Score'],
                row['Lib RT']])

    model_bounding_boxes = pd.DataFrame(model_bounding_boxes)

    model_bounding_boxes.to_csv(
        os.path.join(out_dir, results_csv),
        index=False,
        header=False)

if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-data_dir', '--data_dir', type=str, default='OpenSWATHAutoAnnotated')
    parser.add_argument(
        '-extra_dir', '--extra_dir', type=str, default=None)
    parser.add_argument(
        '-chromatograms_csv',
        '--chromatograms_csv',
        type=str,
        default='chromatograms.csv')
    parser.add_argument(
        '-labels_npy',
        '--labels_npy',
        type=str,
        default=None)
    parser.add_argument(
        '-model_pth',
        '--model_pth',
        type=str,
        default='custom_3_layer_21_kernel_osw_points_wrt_model_150.pth')
    parser.add_argument(
        '-out_dir', '--out_dir', type=str, default='evaluation_results')
    parser.add_argument(
        '-results_csv',
        '--results_csv',
        type=str,
        default='evaluation_results.csv')
    parser.add_argument('-threshold', '--threshold', type=float, default=0.5)
    parser.add_argument('-device', '--device', type=str, default='cpu')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=32)
    parser.add_argument(
        '-load_npy',
        '--load_npy',
        action='store_true',
        default=False)
    parser.add_argument(
        '-npy_dir', '--npy_dir', type=str, default='evaluation_results')
    parser.add_argument(
        '-npy_name', '--npy_name', type=str, default='output_array')
    parser.add_argument('-mode', '--mode', type=str, default='inference')
    parser.add_argument('-num_points', '--num_points', type=int, default=3)
    args = parser.parse_args()

    print(args)

    dataset = ChromatogramsDataset(
        args.data_dir,
        args.chromatograms_csv,
        labels=args.labels_npy,
        extra_path=args.extra_dir,
        transform=ToTensor())

    model = torch.load(args.model_pth, map_location=args.device)
    
    if args.mode == 'rpn':
        model.mode = 'test'
        model.pre_nms_topN = 1
        model.post_nms_topN = 1

    model.eval()

    if args.mode == 'rpn':
        create_rpn_results_file(
            dataset,
            model,
            args.batch_size,
            args.device,
            args.data_dir,
            args.chromatograms_csv,
            args.out_dir,
            args.results_csv)
    else:
        output_array = create_output_array(
            dataset,
            model,
            args.batch_size,
            args.device,
            args.load_npy,
            args.npy_dir,
            args.npy_name,
            args.mode,
            args.threshold)

        if args.mode == 'inference':
            create_results_file(
                output_array,
                args.threshold,
                args.data_dir,
                args.chromatograms_csv,
                args.out_dir,
                args.results_csv)
        elif args.mode == 'stats':
            create_stats_eval_file(
                output_array,
                args.num_points,
                args.data_dir,
                args.chromatograms_csv,
                args.out_dir,
                args.results_csv)
        elif args.mode == 'semisupervised':
            create_semisupervised_results_file(
                output_array,
                args.data_dir,
                args.chromatograms_csv,
                args.out_dir,
                args.results_csv)

    print('It took {0:0.1f} seconds'.format(time.time() - start))
