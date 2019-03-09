import importlib.machinery as imp
import logging
import os
import pprint
import sys
import yaml

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

def run_experiment(yaml_filepath):
    """Example."""
    cfg = load_cfg(yaml_filepath)

    # Print the configuration to verify params
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)

    # Here is an example how you load modules of which you put the path in the
    # configuration. 
    dpath = cfg['dataset']['script_path']
    sys.path.insert(1, os.path.dirname(dpath))
    data = imp.SourceFileLoader('data', cfg['dataset']['script_path']).load_module()

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
