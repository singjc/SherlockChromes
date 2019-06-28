import argparse
import numpy as np
import os
import pandas as pd
import sys
import torch

sys.path.insert(0, '../datasets')
sys.path.insert(0, '../models')

from chromatograms_dataset import ChromatogramsDataset

def create_results_file(
    dataset,
    model,
    data_dir='OpenSWATHAutoAnnotated',
    chromatograms_csv='chromatograms.csv',
    out_dir='.',
    results_csv='evaluation_results.csv',
    threshold=0.5,
    device='cpu'):
    chromatograms = pd.read_csv(os.path.join(
        data_dir, chromatograms_csv))

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

        chromatogram, _ = dataset[i]
        dims = chromatogram.shape

        output = model(torch.from_numpy(
                    np.asarray(
                        chromatogram)).view(1, *dims).float().to(device))[0]
        output = output.detach().to('cpu').numpy()

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', '--data_dir', type=str, default='OpenSWATHAutoAnnotated')
    parser.add_argument('-chromatograms_csv', '--chromatograms_csv', type=str, default='chromatograms.csv')
    parser.add_argument('-labels_npy', '--labels_npy', type=str, default='osw_point_labels.npy')
    parser.add_argument('-model_pth', '--model_pth', type=str, default='custom_3_layer_21_kernel_osw_points_wrt_model_150.pth')
    parser.add_argument('-out_dir', '--out_dir', type=str, default='evaluation_results')
    parser.add_argument('-results_csv', '--results_csv', type=str, default='evaluation_results.csv')
    parser.add_argument('-threshold', '--threshold', type=float, default=0.5)
    parser.add_argument('-device', '--device', type=str, default='cpu')
    args = parser.parse_args()

    dataset = ChromatogramsDataset(
        args.data_dir,
        args.chromatograms_csv,
        args.labels_npy)

    model = torch.load(args.model_pth, map_location=args.device)
    model.eval()

    threshold = args.threshold
    modelname = args.model_pth.split('/')[-1]

    create_results_file(
        dataset,
        model,
        args.data_dir,
        args.chromatograms_csv,
        args.out_dir,
        args.results_csv,
        args.threshold)
