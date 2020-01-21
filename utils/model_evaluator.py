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

from chromatograms_dataset import TarChromatogramsDataset
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
    threshold=0.5):
    output_array = []

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

        output_array.append(output.detach().to('cpu').numpy())

    output_array = np.vstack(output_array)

    if len(output_array.shape) > 2:
        output_array = output_array.reshape(output_array.shape[0], -1)

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
                'Lib RT',
                'Window Size'
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
                    row['Lib RT'],
                    row['Window Size']])
            
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
    npy_name='output_array',
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
                'Lib RT',
                'Window Size',
                'High Quality'
            ]
        ]

    label_output_array = np.zeros((output_array.shape))
    binarized_output_array = np.where(output_array >= threshold, 1, 0)

    for i in range(len(chromatograms)):
        print(i)

        row = chromatograms.iloc[i]

        output = output_array[i]
        binarized_output = binarized_output_array[i]

        left_width, right_width = None, None

        score = 0.0

        high_quality = 0

        regions_of_interest = scipy.ndimage.find_objects(
            scipy.ndimage.label(binarized_output)[0])

        if regions_of_interest:
            scores = [(np.mean(output[r]) + np.max(output[r])) / 2 
                for r 
                in regions_of_interest]
            best_region_idx = np.argmax(scores)
            best_region = regions_of_interest[best_region_idx][0]

            start_idx, end_idx = best_region.start, best_region.stop

            left_width, right_width = start_idx, end_idx

            score = scores[best_region_idx]

            # Find relative index of highest intensity value in region of
            # interest and add to start index to get absolute index
            max_idx = np.argmax(output[regions_of_interest[best_region_idx]])
            max_idx+= start_idx

            if (end_idx - start_idx >= 2 and
                    end_idx - start_idx < 60 and
                    start_idx < max_idx < end_idx and
                    'DECOY_' not in row['Filename']):
                    label_output_array[i, start_idx:end_idx] = np.ones(
                        (end_idx - start_idx)
                    )

                    high_quality = 1

        model_bounding_boxes.append([
                row['ID'],
                row['Filename'],
                row['BB Start'],
                row['BB End'],
                left_width,
                right_width,
                row['OSW Score'],
                score,
                row['Lib RT'],
                row['Window Size'],
                high_quality])

    np.save(os.path.join(out_dir, 'label_' + npy_name), label_output_array)

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
                'Lib RT',
                'Window Size'
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
                row['Lib RT'],
                row['Window Size']])

    model_bounding_boxes = pd.DataFrame(model_bounding_boxes)

    model_bounding_boxes.to_csv(
        os.path.join(out_dir, results_csv),
        index=False,
        header=False)

if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-data_dir', '--data_dir', type=str, default='.')
    parser.add_argument(
        '-dataset', '--dataset', type=str, default='hroest_Strep_600s_175pts.tar')
    parser.add_argument(
        '-extra_dir', '--extra_dir', type=str, default=None)
    parser.add_argument(
        '-chromatograms_csv',
        '--chromatograms_csv',
        type=str,
        default='chromatograms_scored.csv')
    parser.add_argument(
        '-tar_shape',
        '--tar_shape',
        type=str,
        default='6,175')
    parser.add_argument(
        '-model_pth',
        '--model_pth',
        type=str,
        default='/home/xuleon1/projects/def-hroest/xuleon1/SherlockChromes/data/output/round_2_dacat_base_big_decoder_mv_5xlr/dacat_model_41_loss=0.004588034.pth')
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
    parser.add_argument(
        '-preload_path',
        '--preload_path',
        type=str,
        default='hroest_Strep_600s_175pts.npy')
    args = parser.parse_args()

    args.tar_shape = [int(x) for x in args.tar_shape.split(',')]

    print(args)

    preload = args.load_npy == False

    dataset = TarChromatogramsDataset(
        args.data_dir,
        args.chromatograms_csv,
        args.dataset,
        tar_shape=args.tar_shape,
        labels=None,
        preload=preload,
        preload_path=args.preload_path,
        transform=ToTensor())

    print('Data loaded in {0:0.1f} seconds'.format(time.time() - start))

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
            args.threshold)

        if args.mode == 'inference':
            create_results_file(
                output_array,
                args.threshold,
                args.data_dir,
                args.chromatograms_csv,
                args.out_dir,
                args.npy_name,
                args.results_csv)
        elif args.mode == 'stats':
            create_stats_eval_file(
                output_array,
                args.num_points,
                args.data_dir,
                args.chromatograms_csv,
                args.out_dir,
                args.results_csv)

    print('It took {0:0.1f} seconds'.format(time.time() - start))
