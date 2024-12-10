import argparse
import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Tuple, Union, cast, List

from omegaconf import OmegaConf
import os

DataClass = Any
DataClassType = Any


@dataclass
class ConfigBase:
    """Base class that should handle parsing from command line,
    json, dicts.
    """

    @classmethod
    def parse_from_command_line(cls):
        return omegaconf_parse(cls)

    @classmethod
    def parse_from_file(cls, path: str):
        oc = OmegaConf.load(path)
        return cls.parse_from_dict(OmegaConf.to_container(oc))

    @classmethod
    def parse_from_command_line_deprecated(cls):
        result = DataclassArgParser(
            cls, fromfile_prefix_chars="@"
        ).parse_args_into_dataclasses()
        if len(result) > 1:
            raise RuntimeError(
                f"The following arguments were not recognized: {result[1:]}"
            )
        return result[0]

    @classmethod
    def parse_from_dict(cls, inputs):
        return DataclassArgParser._populate_dataclass_from_dict(cls, inputs.copy())

    @classmethod
    def parse_from_flat_dict(cls, inputs):
        return DataclassArgParser._populate_dataclass_from_flat_dict(cls, inputs.copy())

    def save(self, path: str):
        with open(path, "w") as f:
            OmegaConf.save(config=self, f=f)

def parse_args():
    print("Parsing arguments ...")
    parser = argparse.ArgumentParser(description='Arguments for dynamics learning')

    # architecture
    parser.add_argument('--predictor_type',        type=str,       default='mlp')               # mlp, gru, lstm, tcn
    parser.add_argument('--encoder_type',          type=str,       default='cnn')               # cnn, vit
    parser.add_argument('--sim_coeff',             type=float,     default=1.0)
    parser.add_argument('--std_coeff',             type=float,     default=1.0)
    parser.add_argument('--cov_coeff',             type=float,     default=1.0)
    parser.add_argument('--momentum',              type=float,     default=0.99)
    parser.add_argument('--loss_type',             type=str,       default='vicreg')            # vicreg, byol
    parser.add_argument('--norm_features',         type=bool,      default=False)

    ## Transformer Encoder specific arguments
    parser.add_argument('--num_layers',            type=int,       default=2)
    parser.add_argument('--num_heads',             type=int,       default=8)
    parser.add_argument('--d_model',               type=int,       default=64)
    parser.add_argument('--dim_feedforward',       type=int,       default=128)
    parser.add_argument('--dropout',               type=float,     default=0.1)
    
    # training
    parser.add_argument('-r', '--run_id',          type=int,      default=1)
    parser.add_argument('-d', '--gpu_id',          type=int,      default=0)
    parser.add_argument('--num_devices',           type=int,      default=1)
    parser.add_argument('-e', '--epochs',          type=int,      default=20)
    parser.add_argument('-b', '--batch_size',      type=int,      default=128)
    parser.add_argument('-s', '--shuffle',         type=bool,     default=False)
    parser.add_argument('-n', '--num_workers',     type=int,      default=4)
    parser.add_argument('--seed',                  type=int,      default=10)

    # Optimizer
    parser.add_argument('-l', '--learning_rate',   type=float,    default=1e-3)
    parser.add_argument('--warmup_lr',             type=float,    default=1e-2)
    parser.add_argument('--weight_decay',          type=float,    default=1e-4)
    parser.add_argument('--adam_beta1',            type=float,    default=0.9)
    parser.add_argument('--adam_beta2',            type=float,    default=0.999)
    parser.add_argument('--adam_eps',              type=float,    default=1e-08)


    # Experiments
    parser.add_argument('--data_path',             type=str,      default='/scratch/DL24FA', required=True)
    parser.add_argument('--experiment_name',        type=str,      default=None, required=True)
    parser.add_argument('--eval',                  action='store_true')


    args = parser.parse_args()
    for arg in vars(args):
        value = getattr(args, arg)
        if isinstance(value, list):
            setattr(args, arg, [int(e) for e in ''.join(value).split(',')])
    return args

def save_args(args, file_path):
    print("Saving arguments ...")
    with open(file_path, "w") as f:
        for arg in vars(args):
            arg_name = arg
            arg_type = str(type(getattr(args, arg))).replace('<class \'', '')[:-2]
            arg_value = str(getattr(args, arg))
            f.write(arg_name)
            f.write(";")
            f.write(arg_type)
            f.write(";")
            f.write(arg_value)
            f.write("\n")

def load_args(file_path):
    print("Loading arguments ...")
    parser = argparse.ArgumentParser(description='Arguments for unsupervised keypoint extractor')
    with open(file_path, "r") as f:
        for arg in f.readlines():
            arg_name = arg.split(';')[0]
            arg_type = arg.split(';')[1]
            arg_value = arg.split(';')[2].replace('\n', '')
            if arg_type == "str":
                parser.add_argument("--" + arg_name, type=str, default=arg_value)
            elif arg_type == "int":
                parser.add_argument("--" + arg_name, type=int, default=arg_value)
            elif arg_type == "float":
                parser.add_argument("--" + arg_name, type=float, default=arg_value)
            elif arg_type == "list":
                arg_value = [int(e) for e in arg_value[1:-1].split(', ')]
                parser.add_argument("--" + arg_name, type=list, default=arg_value)
            elif arg_type == "tuple":
                arg_value = [int(e) for e in arg_value[1:-1].split(', ')]
                parser.add_argument("--" + arg_name, type=tuple, default=arg_value)
            elif arg_type == "bool":
                arg_value = True if arg_value == "True" else False
                parser.add_argument("--" + arg_name, type=bool, default=arg_value)

    return parser.parse_args()

def check_folder_paths(folder_paths):
    for path in folder_paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Creating folder", path, "...")

