import argparse
import numpy as np
import os
import pandas as pd
import sys
import time
import torch

from torch.utils.data import DataLoader

sys.path.insert(0, '../datasets')
sys.path.insert(0, '../models')

from chromatograms_dataset import ChromatogramsDataset

def create_output_array(
    dataset,
    model,
    batch_size=32,
    device='cpu',
    load_npy=False,
    npy_dir='.',
    npy_name='output_array'):
    output_array = None

    if load_npy:
        output_array = np.load(os.path.join(npy_dir, npy_name + '.npy'))

        return output_array
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in dataloader:
        chromatograms = torch.from_numpy(
            np.asarray(batch[0])).float().to(device)

        output = model(chromatograms)

        if type(output_array) == type(None):
            output_array = output.detach().to('cpu').numpy()
        else:
            output_array = np.vstack(
                (output_array, output.detach().to('cpu').numpy()))

    np.save(os.path.join(npy_dir, npy_name), output_array)

    return output_array

def create_results_file(
    output_array,
    threshold=0.5,
    device='cpu',
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
                'Model Score'
            ]
        ]

    for i in range(len(chromatograms)):
        print(i)

        row = chromatograms.iloc[i]

        output = output_array[i, :]

        largest_idx = np.argmax(output)

        if output[largest_idx] >= threshold:
            start_idx, end_idx = largest_idx, largest_idx

            while output[start_idx - 1] >= threshold:
                start_idx-= 1
            
            while output[end_idx + 1] >= threshold:
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
                str(output[largest_idx])])

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
        '-chromatograms_csv',
        '--chromatograms_csv',
        type=str,
        default='chromatograms.csv')
    parser.add_argument(
        '-labels_npy',
        '--labels_npy',
        type=str,
        default='osw_point_labels.npy')
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
    args = parser.parse_args()

    print(args)

    dataset = ChromatogramsDataset(
        args.data_dir,
        args.chromatograms_csv,
        args.labels_npy)

    model = torch.load(args.model_pth, map_location=args.device)
    model.eval()

    output_array = create_output_array(
        dataset,
        model,
        args.batch_size,
        args.device,
        args.load_npy,
        args.npy_dir,
        args.npy_name)

    create_results_file(
        output_array,
        args.threshold,
        args.device,
        args.data_dir,
        args.chromatograms_csv,
        args.out_dir,
        args.results_csv)

    print('It took {0:0.1f} seconds'.format(time.time() - start))
