import glob
import sys
import torch

from train.eval_semisupervised import evaluate


def main(
        data,
        sampling_fn,
        collate_fn,
        eval_kwargs,
        device):
    sys.path.insert(0, f"{eval_kwargs['path_to_root']}/models")
    sys.path.insert(0, f"{eval_kwargs['path_to_root']}/models/modelzoo1d")

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
