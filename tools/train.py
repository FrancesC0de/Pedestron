from __future__ import division

import argparse
import os
from mmcv import Config

from mmdet import __version__
from mmdet.datasets import build_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
import torch
import warnings

from voc2coco import convert
from cocosplit import cocosplit

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--convert', action='store_false', help='whether convert to json and split')
    parser.add_argument('--datapath', type=str, default='/home/data/130/')
    parser.add_argument('--jsonpath', type=str, default='/home/data/train/train.json')
    parser.add_argument('--s', type=float, default=0.9)
    parser.add_argument('--trainsplit', type=str, default='/home/data/train/train_split.json')
    parser.add_argument('--testsplit', type=str, default='/home/data/train/test_split.json')
    parser.add_argument('--ckp_path', type=str, default='/project/train/models')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    # convert to json
    if args.convert == True:
        convert(json_file=args.jsonpath, xml_dir=args.datapath)
        # split into train and test
        cocosplit(args.jsonpath, args.s, args.trainsplit , args.testsplit)

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    train_dataset = build_dataset(cfg.data.train)
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:  # This is not ideal, but works if the workflow is not crazy
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=train_dataset.CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = train_dataset.CLASSES
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)

        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=args.validate,
            logger=logger,
            ckp_path=args.ckp_path)


if __name__ == '__main__':
    main()
