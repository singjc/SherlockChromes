import logging
import os
import pprint
import sys
import yaml

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from importlib.util import spec_from_file_location, module_from_spec

# logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
#                     level=logging.DEBUG,
#                     stream=sys.stdout)

def run_experiment(yaml_filepath):
    """Example."""
    cfg = load_cfg(yaml_filepath)

    # Print the configuration to verify params
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)

    # TODO: Figure out how to configure more than just the training loop,
    # currently still have to have a new train file per model/dataset/loss
    # function/optimizer, can only config params of each of the formentioned

    device = cfg['general']['device']

    dataset_kwargs = cfg['dataset']['kwargs']

    model_kwargs = cfg['model']['kwargs']

    loss_kwargs = cfg['loss']['kwargs']

    optimizer_kwargs = cfg['optimizer']['kwargs']

    train_path = cfg['train']['script_path']
    sys.path.insert(1, os.path.dirname(train_path))
    train_spec = spec_from_file_location('train', train_path)
    train_mod = module_from_spec(train_spec)
    train_spec.loader.exec_module(train_mod)
    train_kwargs = cfg['train']['kwargs']

    train_mod.main(
        dataset_kwargs,
        model_kwargs,
        loss_kwargs,
        optimizer_kwargs,
        train_kwargs,
        device)

def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Args:
        yaml_filepath (str): Path to yaml config file.

    Output:
        cfg (dict): Dictionary of config params.
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)

    return cfg

def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Args:
        dir_ (str): The absolute path to the directory of script.
        cfg (dict): Dictionary of config params. 

    Output:
        cfg (dict): Dictionary of config params with absolute paths.
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.isfile(cfg[key]):
                logging.error("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])

    return cfg

def get_parser():
    """Get parser object."""
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="experiment definition file",
                        metavar="FILE",
                        required=True)

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    run_experiment(args.filename)
