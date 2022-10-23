import json
import os
import pickle
from collections import Counter, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from gensim.models import FastText
# from nltk.tokenize import sent_tokenize, word_tokenize
# from scipy.linalg import eig
# from skimage.filters import threshold_yen as threshold

from datamodules.unimodal import alphabet


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered."""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


# def cca(views, k=None, eps=1e-12):
#     """Compute (multi-view) CCA
#
#     Args:
#         views (list): list of views where each view `v_i` is of size `N x o_i`
#         k (int): joint projection dimension | if None, find using Otsu
#         eps (float): regulariser [default: 1e-12]
#
#     Returns:
#         correlations: correlations along each of the k dimensions
#         projections: projection matrices for each view
#     """
#     V = len(views)  # number of views
#     N = views[0].size(0)  # number of observations (same across views)
#     os = [v.size(1) for v in views]
#     kmax = np.min(os)
#     ocum = np.cumsum([0] + os)
#     os_sum = sum(os)
#     A, B = np.zeros([os_sum, os_sum]), np.zeros([os_sum, os_sum])
#
#     for i in range(V):
#         v_i = views[i]
#         v_i_bar = v_i - v_i.mean(0).expand_as(v_i)  # centered, N x o_i
#         C_ij = (1.0 / (N - 1)) * torch.mm(v_i_bar.t(), v_i_bar)
#         # A[ocum[i]:ocum[i + 1], ocum[i]:ocum[i + 1]] = C_ij
#         B[ocum[i]:ocum[i + 1], ocum[i]:ocum[i + 1]] = C_ij
#         for j in range(i + 1, V):
#             v_j = views[j]  # N x o_j
#             v_j_bar = v_j - v_j.mean(0).expand_as(v_j)  # centered
#             C_ij = (1.0 / (N - 1)) * torch.mm(v_i_bar.t(), v_j_bar)
#             A[ocum[i]:ocum[i + 1], ocum[j]:ocum[j + 1]] = C_ij
#             A[ocum[j]:ocum[j + 1], ocum[i]:ocum[i + 1]] = C_ij.t()
#
#     A[np.diag_indices_from(A)] += eps
#     B[np.diag_indices_from(B)] += eps
#
#     eigenvalues, eigenvectors = eig(A, B)
#     # TODO: sanity check to see that all eigenvalues are e+0i
#     idx = eigenvalues.argsort()[::-1]  # sort descending
#     eigenvalues = eigenvalues[idx]  # arrange in descending order
#
#     if k is None:
#         t = threshold(eigenvalues.real[:kmax])
#         k = np.abs(np.asarray(eigenvalues.real[0::10]) - t).argmin() * 10  # closest k % 10 == 0 idx
#         print('k unspecified, (auto-)choosing:', k)
#
#     eigenvalues = eigenvalues[idx[:k]]
#     eigenvectors = eigenvectors[:, idx[:k]]
#
#     correlations = torch.from_numpy(eigenvalues.real).type_as(views[0])
#     proj_matrices = torch.split(torch.from_numpy(eigenvectors.real).type_as(views[0]), os)
#
#     return correlations, proj_matrices


# def fetch_emb(lenWindow, minOccur, emb_path, vocab_path, RESET):
#     if not os.path.exists(emb_path) or RESET:
#         with open('../data/cub/text_trainvalclasses.txt', 'r') as file:
#             text = file.read()
#             sentences = sent_tokenize(text)
#
#         texts = []
#         for i, line in enumerate(sentences):
#             words = word_tokenize(line)
#             texts.append(words)
#
#         model = FastText(size=300, window=lenWindow, min_count=minOccur)
#         model.build_vocab(sentences=texts)
#         model.train(sentences=texts, total_examples=len(texts), epochs=10)
#
#         with open(vocab_path, 'rb') as file:
#             vocab = json.load(file)
#
#         i2w = vocab['i2w']
#         base = np.ones((300,), dtype=np.float32)
#         emb = [base * (i - 1) for i in range(3)]
#         for word in list(i2w.values())[3:]:
#             emb.append(model[word])
#
#         emb = np.array(emb)
#         with open(emb_path, 'wb') as file:
#             pickle.dump(emb, file)
#
#     else:
#         with open(emb_path, 'rb') as file:
#             emb = pickle.load(file)
#
#     return emb


# def fetch_weights(weights_path, vocab_path, RESET, a=1e-3):
#     if not os.path.exists(weights_path) or RESET:
#         with open('../data/cub/text_trainvalclasses.txt', 'r') as file:
#             text = file.read()
#             sentences = sent_tokenize(text)
#             occ_register = OrderedCounter()
#
#             for i, line in enumerate(sentences):
#                 words = word_tokenize(line)
#                 occ_register.update(words)
#
#         with open(vocab_path, 'r') as file:
#             vocab = json.load(file)
#         w2i = vocab['w2i']
#         weights = np.zeros(len(w2i))
#         total_occ = sum(list(occ_register.values()))
#         exc_occ = 0
#         for w, occ in occ_register.items():
#             if w in w2i.keys():
#                 weights[w2i[w]] = a / (a + occ / total_occ)
#             else:
#                 exc_occ += occ
#         weights[0] = a / (a + exc_occ / total_occ)
#
#         with open(weights_path, 'wb') as file:
#             pickle.dump(weights, file)
#     else:
#         with open(weights_path, 'rb') as file:
#             weights = pickle.load(file)
#
#     return weights


def fetch_pc(emb, weights, train_loader, pc_path, RESET):
    sentences = torch.cat([d[1][0] for d in train_loader]).int()
    emb_dataset = apply_weights(emb, weights, sentences)

    if not os.path.exists(pc_path) or RESET:
        _, _, V = torch.svd(emb_dataset - emb_dataset.mean(dim=0), some=True)
        v = V[:, 0].unsqueeze(-1)
        u = v.mm(v.t())
        with open(pc_path, 'wb') as file:
            pickle.dump(u, file)
    else:
        with open(pc_path, 'rb') as file:
            u = pickle.load(file)
    return u


def apply_weights(emb, weights, data):
    fn_trun = lambda s: s[:np.where(s == 2)[0][0] + 1] if 2 in s else s
    batch_emb = []
    for sent_i in data:
        emb_stacked = torch.stack([emb[idx] for idx in fn_trun(sent_i)])
        weights_stacked = torch.stack([weights[idx] for idx in fn_trun(sent_i)])
        batch_emb.append(torch.sum(emb_stacked * weights_stacked.unsqueeze(-1), dim=0) / emb_stacked.shape[0])

    return torch.stack(batch_emb, dim=0)


def apply_pc(weighted_emb, u):
    return torch.cat([e - torch.matmul(u, e.unsqueeze(-1)).squeeze() for e in weighted_emb.split(2048, 0)])


class Latent_Classifier(nn.Module):
    """ Generate latent parameters for SVHN image data. """

    def __init__(self, in_n, out_n):
        super(Latent_Classifier, self).__init__()
        self.mlp = nn.Linear(in_n, out_n)

    def forward(self, x):
        return self.mlp(x)


class SVHN_Classifier(nn.Module):
    def __init__(self):
        super(SVHN_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow_hub as hub
from keras import backend as K


class EarlyFusion_Classifier(nn.Module):
    nClasses = 101

    def __init__(self, path):
        super().__init__()

        max_length = 20  # Setup according to the text
        img_width = 299
        img_height = 299

        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)

        model_cnn = models.Sequential()
        model_cnn.add(
            InceptionV3(weights='imagenet', include_top=False, input_tensor=layers.Input(shape=(299, 299, 3))))
        model_cnn.add(layers.AveragePooling2D(pool_size=(8, 8), name='AVG_Pooling'))
        model_cnn.add(layers.Dropout(.4, name='Dropout_0.4'))
        model_cnn.add(layers.Flatten(name='Flatten'))
        model_cnn.add(layers.Dense(128, name='Dense_128'))

        bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
        input_ids = layers.Input(shape=(max_length,), dtype=tf.int32, name="input_word_ids")
        input_masks = layers.Input(shape=(max_length,), dtype=tf.int32, name="input_masks")
        input_segments = layers.Input(shape=(max_length,), dtype=tf.int32, name="segment_ids")
        _, seq_out = bert_layer([input_ids, input_masks, input_segments])
        out = layers.LSTM(128, name='LSTM')(seq_out)
        model_lstm = models.Model([input_ids, input_masks, input_segments], out)

        # Keep the Bert + LSTM layers trainable
        for layer in model_lstm.layers:
            layer.trainable = True

        # Stacking early-fusion multimodal model

        input_word_ids = layers.Input(shape=(max_length,), dtype=tf.int32,
                                      name="input_word_ids")
        input_mask = layers.Input(shape=(max_length,), dtype=tf.int32,
                                  name="input_mask")
        segment_ids = layers.Input(shape=(max_length,), dtype=tf.int32,
                                   name="segment_ids")
        image_input = layers.Input(shape=input_shape, dtype=tf.float32,
                                   name="image")

        image_side = model_cnn(image_input)
        text_side = model_lstm([input_word_ids, input_mask, segment_ids])

        # Concatenate features from images and texts
        merged = layers.Concatenate()([image_side, text_side])
        merged = layers.Dense(256, activation='relu')(merged)
        output = layers.Dense(self.nClasses, activation='softmax', name="class")(merged)

        path = path + '/data/food101/'  # TODO remove
        model = models.Model([input_word_ids, input_mask, segment_ids, image_input], output)
        model.load_weights(f'{path}/early_fusion_weights_0.92.hdf5')

        image_input = layers.Input(shape=(128,), dtype=tf.float32, name="image")
        text_input = layers.Input(shape=(128,), dtype=tf.float32, name="text")

        output = model.layers[6]((image_input, text_input))
        output = model.layers[7](output)
        output = model.layers[8](output)

        model = models.Model([image_input, text_input], output)
        for layer in model.layers:
            layer.trainable = False

        self.model = model

    def forward(self, x1, x2):
        shape = list(x1.size()[:-1]) + [101]
        one_hot = self.model.predict([x1.flatten(end_dim=-2).cpu().numpy(), x2.flatten(end_dim=-2).cpu().numpy()])
        return torch.from_numpy(one_hot).view(shape).to(x1.device)


# Residual block
class ResidualBlockEncoder(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, downsample):
        super(ResidualBlockEncoder, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in)
        self.conv1 = nn.Conv1d(channels_in, channels_in, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(channels_in)
        self.conv2 = nn.Conv1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = residual + 0.3*out
        return out


class ResidualBlockDecoder(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, upsample):
        super(ResidualBlockDecoder, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in)
        self.conv1 = nn.ConvTranspose1d(channels_in, channels_in, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(channels_in)
        self.conv2 = nn.ConvTranspose1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation, output_padding=1)
        self.upsample = upsample

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.upsample:
            residual = self.upsample(x)
        out = 2.0*residual + 0.3*out
        return out


def make_res_block_encoder(channels_in, channels_out, kernelsize, stride, padding, dilation):
    downsample = None
    if (stride != 1) or (channels_in != channels_out) or dilation != 1:
        downsample = nn.Sequential(nn.Conv1d(channels_in, channels_out,
                                             kernel_size=kernelsize,
                                             stride=stride,
                                             padding=padding,
                                             dilation=dilation),
                                   nn.BatchNorm1d(channels_out))
    layers = []
    layers.append(ResidualBlockEncoder(channels_in, channels_out, kernelsize, stride, padding, dilation, downsample))
    return nn.Sequential(*layers)


def make_res_block_decoder(channels_in, channels_out, kernelsize, stride, padding, dilation):
    upsample = None
    if (kernelsize != 1 or stride != 1) or (channels_in != channels_out) or dilation != 1:
        upsample = nn.Sequential(nn.ConvTranspose1d(channels_in, channels_out,
                                                    kernel_size=kernelsize,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    output_padding=1),
                                   nn.BatchNorm1d(channels_out))
    layers = []
    layers.append(ResidualBlockDecoder(channels_in, channels_out, kernelsize, stride, padding, dilation, upsample))
    return nn.Sequential(*layers)


dim = 64
noise = 1e-15
num_features = len(alphabet)


class Text_Classifier(nn.Module):
    def __init__(self, flags):
        super(Text_Classifier, self).__init__()
        self.flags = flags
        self.conv1 = nn.Conv1d(num_features, 2 * dim, kernel_size=1)
        self.resblock_1 = make_res_block_encoder(2 * dim, 3 * dim, kernelsize=4, stride=2, padding=1,
                                                 dilation=1)
        self.resblock_4 = make_res_block_encoder(3 * dim, 2 * dim, kernelsize=4, stride=2, padding=0,
                                                 dilation=1)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.linear = nn.Linear(in_features=2*dim, out_features=10, bias=True) # 10 is the number of classes
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x.transpose(-2,-1)
        h = self.conv1(x)
        h = self.resblock_1(h)
        h = self.resblock_4(h)
        h = self.dropout(h)
        h = h.view(h.size(0), -1)
        h = self.linear(h)
        out = self.sigmoid(h)
        return out


    def get_activations(self, x):
        h = self.conv1(x)
        h = self.resblock_1(h)
        h = self.resblock_2(h)
        h = self.resblock_3(h)
        h = self.resblock_4(h)
        h = h.view(h.size(0), -1)
        return h

