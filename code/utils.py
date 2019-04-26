# -*- coding: utf-8 -*-
import os
import time
import yaml
import logging

import torch


def make_out_paths(args):
    codedir = os.path.split(os.path.realpath(__file__))[0]
    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S')
    data_ciph = args.dataset.replace('.', '_').replace('/', '-')
    if not args.test:
        if not hasattr(args, 'modeldir') or args.modeldir is None:
            args.modeldir = os.path.join(codedir, '../saved_models/', data_ciph + '-' + timestamp)
        if not os.path.exists(args.modeldir):
            os.makedirs(args.modeldir)
        args.logdir = os.path.join(args.modeldir, 'logs/')
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        args.predout = args.modeldir
    else:
        args.testdir = os.path.join(codedir, '../test_results/', data_ciph + '-' + timestamp)
        if not os.path.exists(args.testdir):
            os.makedirs(args.testdir)
        args.logdir = os.path.join(args.testdir, 'logs/')
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        args.predout = args.testdir


def get_logger(args):
    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(args.logdir, 'main.log'))
    ch = logging.StreamHandler()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        fh.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        fh.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_model_conf(model_dir, args, keys):
    with open(os.path.join(model_dir, 'conf.yaml'), 'w') as f:
        yaml_obj = {}
        for key in keys:
            yaml_obj[key] = getattr(args, key)
        yaml.dump(yaml_obj, f)


def load_model_conf(model_dir, args):
    with open(os.path.join(model_dir, 'conf.yaml')) as f:
        conf = yaml.load(f.read(), Loader=yaml.SafeLoader)
        for key in conf:
            setattr(args, key, conf[key])


def load_model(model_dir, model):
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model')))


def save_model(model_dir, model):
    torch.save(model.state_dict(), os.path.join(model_dir, 'model'))
