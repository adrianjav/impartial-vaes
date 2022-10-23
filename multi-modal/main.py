import argparse
import datetime
import sys
import yaml
from pathlib import Path
from tempfile import mkdtemp
from contextlib import redirect_stdout, redirect_stderr

import csv
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks

import models
import objectives_mtl as objectives

from datamodules import DigitsDataModule
from mlsuite import is_interactive_shell, TeeFile


def test(model, test_loader, t_objective, runPath, epoch, plot):
    model.eval()
    model.to('cuda')
    b_loss = 0
    with torch.no_grad():
        for i, dataT in enumerate(test_loader):
            # data = unpack_data(dataT, device=device)
            data = dataT[:-1]
            loss = -t_objective(model, data, K=args.K)
            b_loss += loss.item()
            if i == 0 and plot:
                model.reconstruct(data, runPath, epoch)
                model.analyse(data, runPath, epoch)
                model.generate(runPath, epoch)
    print('====>             Test loss: {:.4f}'.format(b_loss / len(test_loader.dataloader.dataset)))


@torch.no_grad()
def estimate_log_likelihood(model, dataloader, device):
    model.eval()
    model = model.to(device)

    lls = []
    for batch in dataloader:
        lls.append(model.log_likelihoods(batch, mc_samples=10))

    lls = torch.stack(lls[:-1], dim=0).mean(dim=0)  # Last one does not have the same number of elements
    print('Likelihood matrix:')
    print(lls.cpu().numpy())

    print('Actual likelihood')
    print(lls.sum(dim=1).mean(dim=0).item())


def validate(args):
    if args.methods is None:
        args.methods = []

    # load args from disk if pretrained model path is given
    if args.pre_trained:
        args = torch.load(args.pre_trained + '/args.rar')

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # MTL
    if args.mtl_on == '':
        args.methods = []

    mtl_on = []
    args.disable_for_loops = False
    args.disable_q = True

    if 'p' in args.mtl_on:
        mtl_on.append('encoder')
    if 'q' in args.mtl_on:
        args.disable_q = False
    if 's' in args.mtl_on:
        assert not args.sample
        mtl_on.append('decoder')
    args.mtl_on = mtl_on

    assert 'mnist' in args.dataset

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

    csv_file = filename
    with open(csv_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerow(values)


def main(args):
    args = validate(args)
    pl.seed_everything(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    # load model
    modelC = getattr(models, 'VAE_{}'.format(args.dataset))
    model = modelC(args).to(device)

    # if pretrained_path:
    #     print('Loading model {} from {}'.format(model.modelName, pretrained_path))
    #     model.load_state_dict(torch.load(pretrained_path + '/model.rar'))
    #     model._pz_params = model._pz_params

    if not args.experiment:
        args.experiment = model.modelName

    # set up run path
    obj_name = ('m_' if hasattr(model, 'vaes') else '') + args.obj + ('_stl' if args.stl else '') \
               + ('_looser' if args.looser else '') + ('_sample' if args.sample else '') + '_mtl'
    if len(args.methods) > 0:
        method_name = '_'.join(args.methods)
        if 'gradnorm' in args.methods:
            method_name += '_alpha_' + str(args.alpha)
    else:
        method_name = 'vanilla'

    runId = obj_name + '_' + method_name + '_' + datetime.datetime.now().isoformat()
    experiment_dir = Path(args.experiment_path + '/experiments/' + args.experiment + '/seed_' + str(args.seed))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))

    print(args)
    print('Expt:', runPath)
    print('RunID:', runId)

    with open(f'{runPath}/stdout.txt', 'a') as out, open(f'{runPath}/stderr.txt', 'a') as err:
        if is_interactive_shell() and not args.silent:
            out_file = TeeFile(out, sys.stdout)
            err_file = TeeFile(err, sys.stderr)
        else:
            out_file, err_file = out, err

        with redirect_stdout(out_file), redirect_stderr(err_file):
            # save args to run
            with open('{}/args.yml'.format(runPath), 'w') as file:
                yaml.safe_dump(args.__dict__, file)
            torch.save(args, '{}/args.rar'.format(runPath))

            # preparation for training
            modalities = ('mnist', 'svhn', 'text') if 'text' in args.dataset else ('mnist', 'svhn')
            dm = DigitsDataModule(args.experiment_path + '/data', modalities, args.batch_size,
                                    seed=args.seed, dm=20, device=device)
            dm.prepare_data()
            dm.setup('fit')

            t_objective = getattr(objectives, ('m_' if hasattr(model, 'vaes') else '') + 'iwae')

            tb_logger = None
            if not args.no_analytics:
                tb_logger = pl_loggers.TensorBoardLogger(f'{runPath}/tb_logs')

            timer = pl_callbacks.Timer()
            checkpoint_callback = pl_callbacks.ModelCheckpoint(dirpath=runPath, filename='best',
                                                               monitor='validation/nll', save_last=True)

            kwargs = {'gpus': [0]} if not args.no_cuda else {}

            trainer = pl.Trainer(
                max_epochs=args.epochs, logger=tb_logger, default_root_dir=runPath,
                callbacks=[timer, checkpoint_callback], enable_progress_bar=not args.silent,
                **kwargs
            )

            trainer.fit(model, dm)

            seconds = timer.time_elapsed('train')
            print(f'Training finished in {int(seconds)}s ({datetime.timedelta(seconds=seconds)}).')

            estimate_log_likelihood(model, dm.val_dataloader(), device)

            dm.setup('test')

            if not args.no_test:
                metrics = trainer.test(model, datamodule=dm)[0]
                seconds = timer.time_elapsed('test')
                print(f'Testing finished in {int(seconds)}s ({datetime.timedelta(seconds=seconds)}).')
                print('metrics')
                print(metrics)
                write_csv(model, metrics, runPath + '/metrics.last.csv')

                metrics = trainer.test(model, datamodule=dm, ckpt_path='best')[0]
                seconds = timer.time_elapsed('test')
                print(f'Testing finished in {int(seconds)}s ({datetime.timedelta(seconds=seconds)}).')
                print('metrics')
                print(metrics)
                write_csv(model, metrics, runPath + '/metrics.best.csv')

            # metrics = trainer.test(model, dm.test_dataloader())
            # print('metrics')
            # print(metrics)
            test_loader = dm.test_dataloader()
            test(model, test_loader, t_objective, runPath, args.epochs-1, plot='mnist' in args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Modal VAEs')

    parser.add_argument('--experiment', type=str, default='', metavar='E', help='experiment name')
    parser.add_argument('--experiment-path', type=str, default='.', metavar='E', help='experiment path')
    parser.add_argument('--silent', action='store_true', default=False, help='Do not output to sysout.')
    parser.add_argument('--model', type=str, required=True, metavar='M',
                        choices=['mvae', 'mmvae', 'mopoe'], help='model name')
    parser.add_argument('--dataset', type=str, default='mnist_svhn', metavar='M',
                        choices=[s[4:] for s in dir(models) if 'VAE_' in s], help='dataset name (default: mnist_svhn)')
    parser.add_argument('--obj', type=str, default='elbo', metavar='O', choices=['elbo', 'iwae'],
                        help='objective to use (default: elbo)')
    parser.add_argument('--K', type=int, default=20, metavar='K',
                        help='number of particles to use for iwae/dreg (default: 10)')

    parser.add_argument('--looser', action='store_true', default=False,
                        help='use the looser version of IWAE/DREG')
    parser.add_argument('--sample', action='store_true',
                        help='Sample z instead of stratifying (using gumbel + straight through)')
    parser.add_argument('--stl', action='store_true', help='Use STL gradient estimator (pre: dreg)')

    parser.add_argument('--llik_scaling', type=float, default=0.,
                        help='likelihood scaling for cub images/svhn modality when running in'
                             'multimodal setting, set as 0 to use default value')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='batch size for data (default: 256)')
    parser.add_argument('--epochs', type=int, default=10, metavar='E',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--latent-dim', type=int, default=20, metavar='L',
                        help='latent dimensionality (default: 20)')
    parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H',
                        help='number of hidden layers in enc and dec (default: 1)')
    parser.add_argument('--pre-trained', type=str, default="",
                        help='path to pre-trained model (train from scratch if empty)')
    parser.add_argument('--learn-prior', action='store_true', default=False,  # DISABLED
                        help='learn model prior parameters')
    parser.add_argument('--logp', action='store_true', default=False,
                        help='estimate tight marginal likelihood on completion')
    parser.add_argument('--print-freq', type=int, default=0, metavar='f',
                        help='frequency with which to print stats (default: 0)')
    parser.add_argument('--no-analytics', action='store_true', default=False,
                        help='disable plotting analytics')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA use')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-test', action='store_true', default=False, help='Do not test.')

    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')

    group = parser.add_argument_group('mtl')
    group.add_argument('-methods', type=str, nargs='*', choices=['pcgrad', 'gradnorm', 'mgda', 'graddrop', 'nsgd',
                                                                 'imtl-g', 'imtl-l', 'gradvac', 'cagrad'])
    group.add_argument('-alpha', type=float, help='GradNorm\'s alpha hyperparameter')
    group.add_argument('-update-at', type=int, default=20, help='When to update the initial grads of Gradnorm/NSGD ')
    group.add_argument('-mtl-learning-rate', type=float, default=0.001, help='MTL optimizer\'s initial learning rate')
    group.add_argument('-gradvac-decay', type=float, default=0.01)

    # group.add_argument('-mtl-on', type=str, nargs='*', default='all', choices=['all', 'encoder', 'decoder'])
    # group.add_argument('-disable-for-loops', action='store_true',
    #                    help='Disables for MOO loops (useful when not using STL)')
    # group.add_argument('-disable-q', action='store_true', help='Disables the MOO loop for q')
    group.add_argument('-mtl-on', type=str, default='', choices=['', 'p', 'pq', 'pqs'])

    # args
    args = parser.parse_args()

    main(args)
