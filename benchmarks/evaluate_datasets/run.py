import argparse
import os
import sys
import random
from mmengine.config import Config, DictAction

import opencompass
from opencompass.cli.main import main as opencompass_main
from opencompass.utils import run
from opencompass.utils.run import match_cfg_file
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run an evaluation task on vllm')
    # Add shortcut parameters (models, datasets and summarizer)
    parser.add_argument('--models', nargs='+', help='', default=None)
    parser.add_argument('--datasets', nargs='+', help='', default=None)
    parser.add_argument(
        '--config-dir',
        default=os.path.join(ROOT_PATH, 'configs'),
        help='Use the custom config directory instead of config/ to '
        'search the configs for datasets, models and summarizers',
        type=str)
    parser.add_argument('--data-dir',
                        type=str,
                        default="",
                        help="if need override data dir for config")
    parser.add_argument('--max-partition-size',
                        help='The maximum size of an infer task. Only '
                        'effective when "infer" is missing from the config.',
                        type=int,
                        default=40000),
    parser.add_argument('--gen-task-coef',
                        help='The dataset cost measurement coefficient for generation tasks, '
                        'Only effective when "infer" is missing from the config.',
                        type=int,
                        default=20)
    parser.add_argument('--dump-eval-details',
                        help='Whether to dump the evaluation details, including the '
                        'correctness of each sample, bpb, etc.',
                        action='store_true',
                        )
    parser.add_argument('--device', type=str, default="gcu")

    vllm_model_parser = parser.add_argument_group('vllm_model_args')
    parse_vllm_model_args(vllm_model_parser)

    args, remains = parser.parse_known_args()

    if args.models is None:
        assert args.vllm_path is not None, f"either model config or vllm-path should be set"

    return args, remains


def parse_vllm_model_args(vllm_model_parser):
    vllm_model_parser.add_argument('--vllm-path', type=str, default="")
    vllm_model_parser.add_argument('--max-seq-len', type=int, default=4096)
    vllm_model_parser.add_argument('--model-kwargs',
                                   nargs='+',
                                   action=DictAction,
                                   default={})
    vllm_model_parser.add_argument('--generation-kwargs',
                                   nargs='+',
                                   action=DictAction,
                                   default={})
    vllm_model_parser.add_argument('--temperature', type=float, default=0)
    vllm_model_parser.add_argument('--end-str', type=str, default=None)
    vllm_model_parser.add_argument('--max-out-len', type=int, default=100)
    vllm_model_parser.add_argument('--batch-size', type=int, default=32)
    vllm_model_parser.add_argument(
        '--tensor-parallel-size', type=int, default=1)


TEMPLATE = '''
from opencompass.models import VLLM
models = {models}
for i in models:
    i['type'] = VLLM
'''


def gen_config(args, config_name, config_py_path):
    is_ppl = any([i.endswith('ppl') for i in args.datasets])
    path = args.vllm_path
    models = []
    if args.models:
        model_dir = os.path.join(args.config_dir, 'models')
        for model in match_cfg_file(model_dir, args.models):
            cfg = Config.fromfile(model[1])
            if 'models' not in cfg:
                raise ValueError(
                    f'Config file {model[1]} does not contain "models" field')
            if path:
                for m in cfg['models']:
                    m.update(dict(path=path))
            models += cfg['models']
    else:
        abbr = path.rsplit("/")[-1]

        generation_kwargs = args.generation_kwargs
        generation_kwargs.update(dict(temperature=args.temperature))

        model_kwargs = args.model_kwargs
        model_kwargs.update(
            dict(device=args.device,
                 tensor_parallel_size=args.tensor_parallel_size)
        )

        if "gpu_memory_utilization" not in model_kwargs:
            gpu_memory_utilization = 0.7 if is_ppl else 0.9
            model_kwargs.update(
                dict(gpu_memory_utilization=gpu_memory_utilization))

        run_cfg = dict(num_gpus=args.tensor_parallel_size, num_procs=1)

        model = dict(
            abbr=abbr,
            batch_size=args.batch_size,
            generation_kwargs=generation_kwargs,
            max_out_len=args.max_out_len,
            max_seq_len=args.max_seq_len,
            model_kwargs=model_kwargs,
            path=path,
            run_cfg=run_cfg,
        )
        models.append(model)
    with open(config_py_path, 'w') as f:
        f.write(TEMPLATE.format(models=str(models)))
    args.models = [config_name]


def rm_configs(config_py_path):
    try:
        os.remove(config_py_path)
    except OSError as e:
        pass


def get_config_path():
    script_dir = os.path.dirname(os.path.abspath(run.__file__))
    parent_dir = os.path.dirname(script_dir)
    default_configs_dir = os.path.join(parent_dir, 'configs')
    r = random.randint(0, 999999)
    config_name = f'vllm_utils_config_{r:06d}'
    config_py_path = os.path.join(
        default_configs_dir, 'models', f'{config_name}.py')
    return config_name, config_py_path


def main():
    args, remains = parse_args()
    config_name, config_py_path = get_config_path()
    gen_config(args, config_name, config_py_path)
    sys.argv = ['main.py']
    if args.config_dir:
        sys.argv += ['--config-dir', args.config_dir]
    if args.models:
        sys.argv += ['--models'] + args.models
    if args.datasets:
        sys.argv += ['--datasets'] + args.datasets
    if args.data_dir:
        from opencompass.utils.datasets_info import DATASETS_MAPPING
        for i in DATASETS_MAPPING:
            DATASETS_MAPPING[i]['local'] = os.path.abspath(args.data_dir)
    sys.argv += ['--hf-num-gpus', str(args.tensor_parallel_size), '--debug']
    sys.argv += remains
    if not os.getenv("CUDA_VISIBLE_DEVICES", None):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            map(str, range(args.tensor_parallel_size)))

    opencompass_main()
    rm_configs(config_py_path)


if __name__ == '__main__':
    main()
