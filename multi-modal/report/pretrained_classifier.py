import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        s0 = self.s0 = 7
        nf = self.nf = 64
        nf_max = self.nf_max = 1024
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0*s0*s0, 10)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0*self.s0*self.s0)
        out = self.fc(actvn(out))
        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


if __name__ == '__main__':
    import os
    import torch
    from datamodules import PolyMNISTDataModule

    print(os.getcwd())
    device = torch.device('cuda')
    dm = PolyMNISTDataModule('../data/', 64, seed=1, device=device)

    dm.prepare_data()
    dm.setup('fit')

    def train_classifiers():
        epochs = 30

        models = [DigitClassifier().to(device) for _ in range(5)]
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([{'params': m.parameters()} for m in models], lr=0.001)

        for epoch in range(epochs):  # loop over the dataset multiple times
            for step, data in enumerate(dm.train_dataloader()):
                xs, labels = data[:-1], data[-1]

                preds = []
                for i in range(5):
                    pred_i = models[i](xs[i])
                    preds.append(pred_i)

                optimizer.zero_grad()
                losses = [criterion(pred_i, labels) for pred_i in preds]
                if step % 100 == 0:
                    print(epoch, step, [l.item() for l in losses])
                loss = sum(losses)
                loss.backward()
                optimizer.step()

        return models

    models = train_classifiers()

    dm.setup('test')

    acc = [0 for _ in range(5)]
    total = 0
    for data in dm.test_dataloader():
        xs, labels = data[:-1], data[-1]

        for i in range(5):
            output = models[i](xs[i])
            _, predicted = torch.max(output, dim=-1)
            accuracy = (predicted == labels).float()
            acc[i] += accuracy.sum(dim=0).item()
        total += xs[0].size(0)

    acc = [x / total for x in acc]
    print('Test accuracies:', acc)

    for i, m in enumerate(models):
        torch.save(m.state_dict(), f'classifier_polymnist_m{i}.pt')

    print('Finished.')

