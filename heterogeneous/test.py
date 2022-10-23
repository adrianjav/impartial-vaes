import sys
import argparse
import subprocess
import datetime

import torch

import src.plotting as plt
import src.feature_scaling as scaling
from src.datasets import InductiveDataModule, TransductiveDataModule
from src.probabilistc_model import ProbabilisticModel
from src.miscelanea import timed, test_mie_ll

from src.models import VAE, IWAE, DREG, HIVAE

import pytorch_lightning as pl
import numpy as np
import pandas as pd
import seaborn as sns


def validate(args) -> None:
    args.timestamp = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    if args.dataset[-1] == '/':
        args.dataset = args.dataset[:-1]

    dataset = args.dataset
    if dataset[-1] == '/':
        dataset = dataset[:-1]

    args.dataset = args.root + '/' + args.dataset

    # Read types of the dataset
    arguments = ['./read_types.sh', f'{args.dataset}/data_types.csv']
    # if args.trick == 'gamma':
    #     arguments.append('--gamma-trick')
    # elif args.trick == 'bern':
    #     arguments.append('--bern-trick')

    proc = subprocess.Popen(arguments, stdout=subprocess.PIPE)
    out = eval(proc.communicate()[0].decode('ascii'))

    args.probabilistic_model = out['probabilistic model']
    args.categoricals = out['categoricals']


def print_data_info(prob_model, data):
    print()
    print('#' * 20)
    print('Original data')

    x = data
    for i, dist_i in enumerate(prob_model):
        print(f'range of [{i}={dist_i}]: {x[:, i].min()} {x[:, i].max()}')

    print()
    print(f'weights = {[x.item() for x in prob_model.weights]}')
    print()

    print('Scaled data')

    x = prob_model >> data
    for i, dist_i in enumerate(prob_model):
        print(f'range of [{i}={dist_i}]: {x[:, i].min()} {x[:, i].max()}')

    print('#' * 20)
    print()


@torch.no_grad()
def test(model, prob_model, loader, hparams):
    model.eval()
    mask_bc = loader.dataset[:][1].to(hparams.device)
    generated_data = model([loader.dataset[:][0].to(hparams.device), mask_bc, None], mode='reconstruct_sample').cpu()
    gmm_generated_data = model([loader.dataset[:][0].to(hparams.device), mask_bc, None], mode='generate').cpu()

    data = loader.dataset[:][0]
    plt.plot_together([data, generated_data, gmm_generated_data], prob_model, title='',
                      legend=['original', 'reconstructed', 'generated'],
                      path=f'{hparams.path}/marginal_gmm{hparams.n_comp}')

    if hparams.save_pairplot:
        with open(f'{hparams.path}/ground_truth.npy', 'wb') as f:
            np.save(f, data.detach().numpy())
        gt = sns.pairplot(pd.DataFrame(data.detach().numpy()))
        with open(f'{hparams.path}/rec_samples.npy', 'wb') as f:
            np.save(f, generated_data.detach().numpy())
        rec = sns.pairplot(pd.DataFrame(generated_data.detach().numpy()))

        for i in range(len(prob_model)):
            rec.axes[i, 0].set_ylim(gt.axes[i, 0].get_ylim())
            rec.axes[0, i].set_xlim(gt.axes[0, i].get_xlim())

        gt.savefig(f'{hparams.path}/gt_pairplot.png', facecolor='w', edgecolor='w')
        rec.savefig(f'{hparams.path}/rec_pairplot.png', facecolor='w', edgecolor='w')


def main(hparams):
    validate(hparams)
    pl.seed_everything(hparams.seed)

    if hparams.to_file:
        sys.stdout = open(f'{hparams.path}/test_{hparams.test_on}_stdout.txt', 'w')
        sys.stderr = open(f'{hparams.path}/test_{hparams.test_on}_stderr.txt', 'w')

    prob_model = ProbabilisticModel(hparams.probabilistic_model)
    print('Likelihoods:', [str(d) for d in prob_model])
    print('Dataset:', hparams.dataset)

    # preprocess_fn = generate_preprocess_functions(prob_model, args)
    preprocess_fn = [scaling.standardize(prob_model, 'continuous')]
    if hparams.transductive:
        raise NotImplementedError('Transductive settings disabled for now.')
        # dm = TransductiveDataModule(hparams.dataset, hparams.miss_perc, hparams.miss_suffix, hparams.categoricals,
        #                             prob_model, hparams.batch_size, preprocess_fn)
    else:
        dm = InductiveDataModule(hparams.dataset, hparams.miss_perc, hparams.miss_suffix, hparams.categoricals, prob_model,
                                 hparams.batch_size, preprocess_fn)

    dm.prepare_data()
    dm.setup(stage='test')
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()
    val_loader = dm.val_dataloader()
    # tb_logger = None
    # if hparams.tensorboard:
    #     tb_logger = pl_loggers.TensorBoardLogger(f'{hparams.root}/tb_logs')

    # Evaluate
    prob_model = prob_model.to('cpu')

    print('Loading and evaluating best model.')
    model = {
        'vae': VAE, 'iwae': IWAE, 'dreg': DREG, 'hivae': HIVAE
    }[hparams.model].load_from_checkpoint(f'{hparams.path}/{hparams.test_on}.ckpt', prob_model=prob_model)

    train_dataset = train_loader.dataset
    mask_bc = train_dataset[:][1]  # if not missing else dataset[:][1]

    model.fit_expost([train_dataset[:][0].to(hparams.device), mask_bc, None], n_components=hparams.n_comp)
    test(model, prob_model, test_loader, hparams)
    test_mie_ll(model, prob_model, test_loader.dataset, hparams.device)

    if hparams.save_gmm:

        val_dataset = val_loader.dataset
        mask_bc_val = val_dataset[:][1]
        test_dataset = test_loader.dataset
        data = torch.cat((train_dataset[:][0], val_dataset[:][0], test_dataset[:][0]))
        # save the original data after preprocessing
        with open(f'{hparams.path}/original_data.npy', 'wb') as f:
            np.save(f, data.detach().numpy())
        # generate GMM samples of size training_size
        data_tosample = torch.cat((train_dataset[:][0], val_dataset[:][0]))
        mask_bc = torch.cat((mask_bc, mask_bc_val))
        gmm_generated_data = model([data_tosample.to(hparams.device), mask_bc, None], mode='generate').cpu()
        with open(f'{hparams.path}/gmm_samples{hparams.n_comp}.npy', 'wb') as f:
            np.save(f, gmm_generated_data.detach().numpy())


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    # Configuration
    parser = argparse.ArgumentParser('')

    # General
    parser.add_argument('-seed', type=int, default=None)
    parser.add_argument('-device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('-root', type=str, default='.', help='Output folder (default: \'%(default)s)\'')
    parser.add_argument('-to-file', action='store_true', help='Redirect output to \'test_best_stdout.txt\'')
    parser.add_argument('-test-on', type=str, default='best', choices=['best', 'last'])

    parser.add_argument('-transductive', action='store_true', help='Use the transductive setting.')
    parser.add_argument('-model', type=str, required=True, choices=['vae', 'iwae', 'hivae', 'dreg'])

    # Tracking
    parser.add_argument('-tensorboard', action='store_true', help='Activates tensorboard logs.')

    # Dataset
    group = parser.add_argument_group('dataset')
    group.add_argument('-batch-size', type=int, default=1024, help='Batch size (%(default)s)')
    group.add_argument('-dataset', type=str, required=True, help='Dataset to use (path to folder)')
    group.add_argument('-miss-perc', type=int, required=True, help='Missing percentage')
    group.add_argument('-miss-suffix', type=int, required=True, help='Suffix of the missing percentage file')

    parser.add_argument('-path', type=str, required=True, help='Path to the experiment folder')
    group.add_argument('-save-pairplot', action='store_true', help='Save reconstructed and ground truth pairplots')
    # GMM
    group.add_argument('-n-comp', type=int, default=100, help='GMM number of components')
    group.add_argument('-save-gmm', action='store_true', help='Save GMM samples in \'gmm_samples.npy\'')
    args = parser.parse_args()
    main(args)

    sys.exit(0)
