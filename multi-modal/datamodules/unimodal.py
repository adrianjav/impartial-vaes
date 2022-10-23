import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNIST(datasets.MNIST):
    def __init__(self, path: str, train: bool, download: bool = False):
        super().__init__(path, train=train, download=download, transform=transforms.ToTensor())


class SVHN(datasets.SVHN):
    def __init__(self, path: str, train: bool, download: bool = False):
        super().__init__(path, split='train' if train else 'test', download=download, transform=transforms.ToTensor())


alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"\\/|_@#$%^&*~`+-=<>()[]{} \n'
digit_text = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
digit_text_numeric = tuple(tuple(alphabet.find(c) for c in s) for s in digit_text)


def to_one_hot(x, size):
    x_one_hot = x.new_zeros(x.size(0), size)
    x_one_hot.scatter_(1, x.unsqueeze(-1).long(), 1).float()

    return x_one_hot


def create_text_from_label_mnist(size, label, start_index):
    text = digit_text_numeric[label]
    sequence = torch.ones(size) * alphabet.find(' ')

    # start_index = random.randint(0, size - 1 - len(text))
    start_index = int(start_index * (size - 1 - len(text)))
    sequence[start_index:start_index + len(text)] = sequence.new(text)

    sequence_one_hot = to_one_hot(sequence, len(alphabet))
    return sequence_one_hot


class Text(Dataset):
    size: int = 8

    def __init__(self, labels, seed):
        super().__init__()
        self.labels = labels
        self.start_indexes = torch.rand(len(labels), generator=torch.Generator().manual_seed(seed))

    def __getitem__(self, item):
        return create_text_from_label_mnist(self.size, self.labels[item], self.start_indexes[item]), self.labels[item]


if __name__ == '__main__':
    print(max([len(s) for s in digit_text_numeric]))
    print({a: b for a, b in zip(digit_text, digit_text_numeric)})
    print(create_text_from_label_mnist(10, 2).shape)

    dataset = Text([1, 2, 3])
    print(dataset[0])


