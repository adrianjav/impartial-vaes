import csv
import argparse
import datetime
import sys
from contextlib import redirect_stdout, redirect_stderr

import torch
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks

import models

from datamodules import DigitsDataModule, Food101DataModule, PolyMNISTDataModule
from mlsuite import is_interactive_shell, TeeFile


def validate(args, hparams):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.experiment_path = args.root_path
    assert args.seed == hparams.seed
    return args


def write_csv(model, metrics, filename):
    # set up run path
    obj_name = model.hparams.obj + ('_looser' if model.hparams.looser else '') + \
               ('_sample' if model.hparams.sample else '')

    if len(model.hparams.methods) > 0:
        method_name = '_'.join(model.hparams.methods)
        if 'gradnorm' in model.hparams.methods or 'cagrad' in model.hparams.methods:
            method_name += '_alpha_' + str(model.hparams.alpha)
        if 'gradvac' in model.hparams.methods:
            method_name += '_decay_' + str(model.hparams.gradvac_decay)
    else:
        method_name = 'vanilla'

    ass1 = 'encoder' in model.hparams.mtl_on
    ass2 = not model.hparams.disable_q
    ass3 = model.hparams.mtl_on in ['all', 'decoder'] and not model.hparams.sample

    ass1 = ass1 and method_name != 'vanilla'
    ass2 = ass2 and method_name != 'vanilla'
    ass3 = ass3 and method_name != 'vanilla'

    columns = ['model', 'dataset', 'loss', 'method', 'seed', 'ass1', 'ass2', 'ass3'] + list(metrics.keys())
    values = [model.hparams.model, args.dataset, obj_name, method_name, args.seed, ass1, ass2, ass3] + list(
        metrics.values())

    csv_file = args.exp_path + '/' + filename
    with open(csv_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerow(values)


def main(args):
    pl.seed_everything(args.seed)

    # load model
    modelC = getattr(models, 'VAE_{}'.format(args.dataset))
    model = modelC.load_from_checkpoint(f'{args.exp_path}/best.ckpt')

    args = validate(args, model.hparams)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = model.to(device)
    model.hparams.update(vars(args))

    with open(f'{args.exp_path}/test_stdout.txt', 'a') as out, open(f'{args.exp_path}/test_stderr.txt', 'a') as err:
        if is_interactive_shell() and not args.silent:
            out_file = TeeFile(out, sys.stdout)
            err_file = TeeFile(err, sys.stderr)
        else:
            out_file, err_file = out, err

        with redirect_stdout(out_file), redirect_stderr(err_file):
            print(f'Best model loaded.')

            # preparation for training
            if args.dataset == 'food':
                dm = Food101DataModule(args.experiment_path + '/data/food101/', args.batch_size, seed=args.seed)
            elif args.dataset == 'polymnist':
                dm = PolyMNISTDataModule(args.experiment_path, args.batch_size, seed=args.seed)
            else:
                modalities = ('mnist', 'svhn', 'text') if 'text' in args.dataset else ('mnist', 'svhn')
                dm = DigitsDataModule(args.experiment_path + '/data', modalities, args.batch_size,
                                      seed=args.seed, dm=20, device=device)
            dm.prepare_data()
            dm.setup(stage='fit')
            dm.setup(stage='test')

            timer = pl_callbacks.Timer()
            kwargs = {'gpus': [0]} if not args.no_cuda else {}

            trainer = pl.Trainer(
                default_root_dir=args.exp_path, callbacks=[timer], enable_progress_bar=not args.silent, logger=False,
                **kwargs
            )

            # Best test
            metrics = trainer.test(model, datamodule=dm)[0]

            seconds = timer.time_elapsed('test')
            print(f'Testing finished in {int(seconds)}s ({datetime.timedelta(seconds=seconds)}).')

            print('metrics')
            print(metrics)

            write_csv(model, metrics, 'metrics.best.csv')

            # Latest test
            model = modelC.load_from_checkpoint(f'{args.exp_path}/last.ckpt')
            print(f'Last model loaded.')

            metrics = trainer.test(model, datamodule=dm)[0]

            seconds = timer.time_elapsed('test')
            print(f'Testing finished in {int(seconds)}s ({datetime.timedelta(seconds=seconds)}).')

            print('metrics')
            print(metrics)

            write_csv(model, metrics, 'metrics.last.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Modal VAEs')

    parser.add_argument('--root-path', type=str, default='..', metavar='E', help='root path.')
    parser.add_argument('--exp-path', type=str, default='', metavar='E', help='experiment path.')
    parser.add_argument('--test-on', type=str, default='test', choices=['test', 'val'])

    parser.add_argument('--silent', action='store_true', default=False, help='Do not output to sysout.')

    parser.add_argument('--dataset', type=str, default='mnist_svhn', metavar='M',
                        choices=[s[4:] for s in dir(models) if 'VAE_' in s], help='model name (default: mnist_svhn)')

    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='batch size for data (default: 256)')
    parser.add_argument('--no-analytics', action='store_true', default=False,
                        help='disable plotting analytics')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA use')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # args
    args = parser.parse_args()

    main(args)
