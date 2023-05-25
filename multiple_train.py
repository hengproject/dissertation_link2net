from train import run
import argparse
from util.load_util import yaml_plain_loader as config_loader
from os.path import join as path_join
from os import listdir as ls

def multi_config_getter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='the directory path of config.yml', default='./configs')
    args = parser.parse_args()
    config_list = [config for config in ls(args.dir)]
    configs = [config_loader(path_join(args.dir,config)) for config in config_list]
    return configs

if __name__ == '__main__':
    configs = multi_config_getter()
    for i,config in enumerate(configs):
        for _ in range(15):
            print(f'{i}process started=====================================================')
        run(config)