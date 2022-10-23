import os
from typing import Union, Sequence, Optional

import torch
from torch.utils.data import Dataset, Subset, DataLoader, random_split

import pytorch_lightning as pl
from pl_bolts.datamodules import AsynchronousLoader

from .unimodal import MNIST, SVHN, Text


class UnionDataset(Dataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets
        assert all(len(datasets[0]) == len(d) for d in datasets[1:])

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, item):
        output = []
        for dataset in self.datasets:
            output.append(dataset[item][0])
        output.append(self.datasets[-1][item][1])
        return output


def rand_match_on_idx(l1, idx1, l2, idx2, dm, max_d=10000):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    _idx1, _idx2 = [], []
    for l in l1.unique():  # assuming both have same idxs
        l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
        n = min(l_idx1.size(0), l_idx2.size(0), max_d)
        l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
        for _ in range(dm):
            _idx1.append(l_idx1[torch.randperm(n)])
            _idx2.append(l_idx2[torch.randperm(n)])

    return torch.cat(_idx1), torch.cat(_idx2)


class DigitsDataModule(pl.LightningDataModule):
    train_dataset: Subset
    val_dataset: Subset
    test_dataset: Subset

    def __init__(self, dir_path: str, datasets: Sequence[str], batch_size: int, seed: int, dm: int = 20,
                 split: float = 0.9, num_workers: int = 8, device: torch.device = torch.device('cuda')):
        super().__init__()

        if dir_path[-1] == '/':
            dir_path = dir_path[:-1]

        self.dir_path = dir_path
        self.datasets = datasets
        self.batch_size = batch_size
        self.seed = seed
        self.dm = dm
        self.split = split
        self.num_workers = num_workers
        self.device = device

        assert datasets == ('mnist', 'svhn') or datasets == ('mnist', 'svhn', 'text')

    def prepare_data(self):
        assert os.path.exists(self.dir_path) and os.path.isdir(self.dir_path)

    def _setup_step(self, train, split):
        mnist_dataset = MNIST(self.dir_path, train=train, download=True)
        svhn_dataset = SVHN(self.dir_path, train=train, download=True)
        text_dataset = Text(mnist_dataset.targets, self.seed)

        mnist_l, mnist_li = mnist_dataset.targets.sort()
        svhn_l, svhn_li = torch.from_numpy(svhn_dataset.labels).sort()

        mnist_idx, svhn_idx = rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li, max_d=10000, dm=self.dm)

        mnist_dataset = Subset(mnist_dataset, mnist_idx)
        svhn_dataset = Subset(svhn_dataset, svhn_idx)
        text_dataset = Subset(text_dataset, mnist_idx)

        if self.datasets == ('mnist', 'svhn'):
            multimodal_dataset = UnionDataset(mnist_dataset, svhn_dataset)
        else:
            multimodal_dataset = UnionDataset(mnist_dataset, svhn_dataset, text_dataset)

        size = len(multimodal_dataset)
        dataset = Subset(multimodal_dataset, range(size))
        if split == 1.:
            return [dataset]

        split_train = int(size * split)
        splits = (split_train, size - split_train)
        assert all([x > 0 for x in splits])

        generator = torch.Generator().manual_seed(self.seed)
        return random_split(multimodal_dataset, splits, generator=generator)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None:
            self.train_dataset, self.test_dataset = self._setup_step(train=True, split=self.split)
        elif stage == 'fit':
            self.train_dataset, self.val_dataset = self._setup_step(train=True, split=self.split)
        elif stage == 'test':
            self.test_dataset = self._setup_step(train=False, split=1.)[0]
        else:
            raise KeyError

    def train_dataloader(self) -> Union[DataLoader, AsynchronousLoader]:
        if self.device != torch.device('cpu'):
            kwargs = {'num_workers': self.num_workers, 'pin_memory': True}
            return AsynchronousLoader(self.train_dataset, device=self.device, batch_size=self.batch_size, shuffle=True,
                                      **kwargs)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, AsynchronousLoader]:
        if self.device != torch.device('cpu'):
            kwargs = {'num_workers': self.num_workers, 'pin_memory': True}
            return AsynchronousLoader(self.val_dataset, device=self.device, batch_size=self.batch_size, shuffle=True,
                                      **kwargs)
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self) -> Union[DataLoader, AsynchronousLoader]:
        if self.device != torch.device('cpu'):
            kwargs = {'num_workers': self.num_workers, 'pin_memory': True}
            return AsynchronousLoader(self.test_dataset, device=self.device, batch_size=self.batch_size, shuffle=True,
                                      **kwargs)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)


if __name__ == '__main__':
    dm = DigitsDataModule('../../data', ('mnist', 'svhn', 'text'), 32, seed=1, dm=2, device=torch.device('cpu'))

    dm.prepare_data()
    dm.setup('fit')

    loader = dm.train_dataloader()

    a = next(iter(loader))
    print(a[-1])
