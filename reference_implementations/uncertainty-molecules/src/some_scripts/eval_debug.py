#!/usr/bin/env python3
import yaml
from src.some_scripts.standard_fit import run

if __name__ == '__main__':
    with open("src/some_scripts/eval.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    _config = {
        'overwrite': 0,
        'db_collection': 'debug-stuff',
    }
    grid_params = {}
    for k in config['grid']:
        if config['grid'][k]['type'] == 'choice':
            grid_params[k] = config['grid'][k]['options'][0]
        elif config['grid'][k]['type'] == 'range':
            grid_params[k] = config['grid'][k]['min']
    #config['fixed']['debug'] = True
    run(**config['fixed'], **grid_params)