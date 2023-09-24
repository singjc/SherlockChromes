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
            model_filenames = sorted(glob.glob(f'{pattern}'))

            for model_filename in model_filenames:
                model = torch.load(model_filename)

                if hasattr(model, 'get_model'):
                    model = model.get_model()

                if hasattr(model, 'set_output_probs'):
                    model.set_output_probs(True)

                print(f'Loaded {model_filename}')

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
                replace_kwargs = input("Enter 'y' to replace kwarg values: ")

                if replace_kwargs == 'y':
                    replacements = input("CSVs with keyword:new_value pairs: ")

                    for replacement in replacements.split(','):
                        key, val = replacement.split(':')

                        if val.replace('.', '', 1).isdigit():
                            if '.' in val:
                                val = float(val)
                            else:
                                val = int(val)
                        elif val == 'True':
                            val = True
                        elif val == 'False':
                            val = False

                        eval_kwargs[key] = val
    else:
        raise NotImplementedError('Interactive mode only!')
