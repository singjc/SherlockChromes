import glob
import sys
import torch

from train.eval_semisupervised import evaluate

sys.path.insert(0, '..')
sys.path.insert(0, '../models')
sys.path.insert(0, '../models/modelzoo1d')


def main(
        data,
        sampling_fn,
        collate_fn,
        eval_kwargs,
        device):
    if 'mode' in eval_kwargs and eval_kwargs['mode'] == 'interactive':
        loop = True

        while loop:
            pattern = input('Input model filename pattern to glob for: ')
            model_filenames = glob.glob(f'{pattern}')

            for model_filename in model_filenames:
                model = torch.load(model_filename)
                evaluate(
                    data,
                    model,
                    sampling_fn,
                    collate_fn,
                    device,
                    **eval_kwargs)

            end_loop_command = input("Enter 'break' to exit: ")

            if end_loop_command == 'break':
                break
    else:
        evaluate(
            data,
            model,
            sampling_fn,
            collate_fn,
            device,
            **eval_kwargs)
