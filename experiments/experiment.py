import logging
import os
import pprint
import sys
import yaml

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from importlib.util import spec_from_file_location, module_from_spec

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

def run_experiment(yaml_filepath):
    """Example."""
    cfg = load_cfg(yaml_filepath)

    # Print the configuration to verify params
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)

    assert 'dataset' in cfg, 'dataset is required!'
    assert 'model' in cfg, 'model is required!'
    assert 'train' in cfg, 'train is required!'

    if 'general' in cfg:
        device = cfg['general']['device']
    else:
        device = 'cpu'

    if 'transform' in cfg:
        transforms_path = cfg['transform']['script_path']
        transforms_mod_name = cfg['transform']['module_name']
        transforms_spec = spec_from_file_location(
            transforms_mod_name, transforms_path)
        transforms_mod = module_from_spec(transforms_spec)
        transforms_spec.loader.exec_module(transforms_mod)
        sys.modules[transforms_mod_name] = transforms_mod
        transform = getattr(
            transforms_mod, cfg['transform']['transform_name'])()
    else:
        transfrom = None


    dataset_kwargs = cfg['dataset']['kwargs']
    data_path = cfg['dataset']['script_path']
    sys.path.insert(0, data_path)
    data_mod_name = cfg['dataset']['module_name']
    data_spec = spec_from_file_location(data_mod_name, data_path)
    data_mod = module_from_spec(data_spec)
    data_spec.loader.exec_module(data_mod)
    sys.modules[data_mod_name] = data_mod
    dataset = getattr(
        data_mod, cfg['dataset']['dataset_name'])(
            **dataset_kwargs, transform=transform)

    model_kwargs = cfg['model']['kwargs']
    model_path = cfg['model']['script_path']
    model_mod_name = cfg['model']['module_name']
    model_spec = spec_from_file_location(model_mod_name, model_path)
    model_mod = module_from_spec(model_spec)
    model_spec.loader.exec_module(model_mod)
    sys.modules[model_mod_name] = model_mod
    model = getattr(
        model_mod, cfg['model']['model_name'])(**model_kwargs)

    if 'loss' in cfg:
        loss_kwargs = cfg['loss']['kwargs']
        loss_path = cfg['loss']['script_path']
        loss_mod_name = cfg['loss']['module_name']
        loss_spec = spec_from_file_location(loss_mod_name, loss_path)
        loss_mod = module_from_spec(loss_spec)
        loss_spec.loader.exec_module(loss_mod)
        sys.modules[loss_mod_name] = loss_mod
        loss = getattr(
            loss_mod, cfg['loss']['loss_name'])(**loss_kwargs)
    else:
        loss = None

    optimizer_kwargs = cfg['optimizer']['kwargs']

    if 'sampling_fn' in cfg:
        sampling_fns_path = cfg['sampling_fn']['script_path']
        sampling_fns_mod_name = cfg['sampling_fn']['module_name']
        sampling_fns_spec = spec_from_file_location(
            sampling_fns_mod_name, sampling_fns_path)
        sampling_fns_mod = module_from_spec(sampling_fns_spec)
        sampling_fns_spec.loader.exec_module(sampling_fns_mod)
        sys.modules[sampling_fns_mod_name] = sampling_fns_mod
        sampling_fn = getattr(
            sampling_fns_mod, cfg['sampling_fn']['sampling_fn_name'])
    else:
        sampling_fn = None

    if 'collate_fn' in cfg:
        collate_fns_path = cfg['collate_fn']['script_path']
        collate_fns_mod_name = cfg['collate_fn']['module_name']
        collate_fns_spec = spec_from_file_location(
            collate_fns_mod_name, collate_fns_path)
        collate_fns_mod = module_from_spec(collate_fns_spec)
        collate_fns_spec.loader.exec_module(collate_fns_mod)
        sys.modules[collate_fns_mod_name] = collate_fns_mod
        collate_fn = getattr(
            collate_fns_mod, cfg['collate_fn']['collate_fn_name'])
    else:
        collate_fn = None

    train_path = cfg['train']['script_path']
    sys.path.insert(1, os.path.dirname(train_path))
    train_spec = spec_from_file_location('train', train_path)
    train_mod = module_from_spec(train_spec)
    train_spec.loader.exec_module(train_mod)
    train_kwargs = cfg['train']['kwargs']

    train_mod.main(
        dataset,
        model,
        loss,
        sampling_fn,
        collate_fn,
        optimizer_kwargs,
        train_kwargs,
        device)

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
    
def extract_module_names(cfg):
    """
    Extract all module names based on filename from script paths.

    Args:
        cfg (dict): Dictionary of config params. 

    Output:
        cfg (dict): Dictionary of config params with added module names.
    """
    paths = []
    for key1 in cfg.keys():
        for key2 in cfg[key1]:
            if key2.endswith("_path"):
                paths.append((key1, key2))

    for path in paths:
        cfg[path[0]]['module_name'] = os.path.splitext(
            os.path.basename(cfg[path[0]][path[1]]))[0]

    return cfg

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
    cfg = extract_module_names(cfg)

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
