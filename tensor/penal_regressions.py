"""Class and function for penalized regressions with tensorflow."""
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from tensorflow_example import download_data
from sklearn_penal_regression import sklearn_models
from tensorflow_penal_regression import tensorflow_models
import os
import pandas as pd
import numpy as np
import pickle
import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


def hard_sigmoid(x):
    """Hard Sigmoid function."""
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))


class _L0Norm(nn.Module):
    """L0 norm."""

    def __init__(self, origin, loc_mean=0, loc_sdev=0.01,
                 beta=2 / 3, gamma=-0.1,
                 zeta=1.1, fix_temp=True):
        """Class of layers using L0 Norm.

        :param origin: original layer such as nn.Linear(..), nn.Conv2d(..)
        :param loc_mean: mean of the normal distribution which generates initial location parameters
        :param loc_sdev: standard deviation of the normal distribution which generates initial location parameters
        :param beta: initial temperature parameter
        :param gamma: lower bound of "stretched" s
        :param zeta: upper bound of "stretched" s
        :param fix_temp: True if temperature is fixed
        """
        super(_L0Norm, self).__init__()
        self._origin = origin
        self._size = self._origin.weight.size()
        self.loc = nn.Parameter(torch.zeros(self._size).normal_(loc_mean, loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros(self._size))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = np.log(-gamma / zeta)

    def _get_mask(self):
        if self.training:
            self.uniform.uniform_()
            u = Variable(self.uniform)
            s = F.sigmoid((torch.log(u)-torch.log(1-u)+self.loc)/self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = F.sigmoid(self.loc-self.temp*self.gamma_zeta_ratio).sum()
        else:
            s = F.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
            penalty = 0
        return hard_sigmoid(s), penalty


class L0Linear(_L0Norm):
    """Linear model with L0 norm."""

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        """Linear model with L0 norm."""
        super(L0Linear, self).__init__(nn.Linear(in_features, out_features,
                                                 bias=bias), **kwargs)

    def forward(self, input):
        """Forward function with mask and penalty."""
        mask, penalty = self._get_mask()
        out = F.linear(input, self._origin.weight * mask, self._origin.bias)
        m = nn.Sigmoid()
        out = m(out)
        return out, penalty


class _L12Norm(nn.Module):
    """L1 or L2 norm linear model."""

    def __init__(self, origin):
        """L1 or L2 norm linear model."""
        super(_L12Norm, self).__init__()
        self._origin = origin

    def _l1_reg(self):
        if self.training:
            penalty = Variable(torch.FloatTensor(1), requires_grad=True)
            penalty = torch.sum(torch.abs(self._origin.weight))
        else:
            penalty = 0
        return penalty

    def _l2_reg(self):
        if self.training:
            penalty = Variable(torch.FloatTensor(1), requires_grad=True)
            penalty = torch.sum(torch.mul(self._origin.weight,
                                          self._origin.weight))
        else:
            penalty = 0
        return penalty


class L12Linear(_L12Norm):
    """Linear model with L0 norm."""

    def __init__(self, in_features, out_features, penal, bias=True, **kwargs):
        """Linear model with L0 norm."""
        super(L12Linear, self).__init__(nn.Linear(in_features, out_features, bias=bias), **kwargs)
        self.penal = penal

    def forward(self, input):
        """Forward function with mask and penalty."""
        if self.penal == 'l1':
            penalty = self._l1_reg()
            out = F.linear(input, self._origin.weight, self._origin.bias)
        elif self.penal == 'l2':
            penalty = self._l2_reg()
            out = F.linear(input, self._origin.weight, self._origin.bias)
        else:
            raise ValueError('wrong norm specified')
        m = nn.Sigmoid()
        out = m(out)
        return out, penalty


class pytoch_linear(object):
    """Penalized regresssion with L1/L2/L0 norm."""

    def __init__(self, X, y, model_log, overwrite=False):
        """Penalized regression."""
        super(pytoch_linear, self).__init__()
        self.model_log = model_log
        self.overwrite = overwrite
        self.X = X
        self.y = y.reshape(len(y), 1)
        self.input_dim = X.shape[1]
        self.output_dim = 1
        print(self.input_dim, self.output_dim)
        if not os.path.isfile(model_log):
            with open(model_log, 'w', encoding='utf-8') as f:
                pickle.dump([], f)
        elif overwrite:
            with open(model_log, 'wb') as f:
                pickle.dump([], f)
        assert os.path.isfile(self.model_log)

    def run(self, penal='l1', **kwargs):
        """Run regression with the given paramters."""
        epochs = 601
        l_rate = 0.01
        if penal == 'l1':
            model = L12Linear(self.input_dim, self.output_dim, penal=penal)
        elif penal == 'l2':
            model = L12Linear(self.input_dim, self.output_dim, penal=penal)
        elif penal == 'l0':
            model = L0Linear(self.input_dim, self.output_dim, **kwargs)
        else:
            raise ValueError('incorrect norm specified')

        optimizer = torch.optim.SGD(model.parameters(), lr=l_rate)
        for _ in range(epochs):
            input = Variable(torch.from_numpy(self.X)).float()
            labels = Variable(torch.from_numpy(self.y)).float()
            optimizer.zero_grad()
            outputs, penalty = model.forward(input)
            loss = -(labels * torch.log(outputs) + (1-labels)*torch.log(1-outputs)).mean()
            loss = loss + 0.01*penalty
            loss.backward()
            optimizer.step()
            if _ % 100 == 0:
                print("epoch {}, loss {}, norm {}".format(_, loss.item(), penalty.item()))
        predict, penal = model.forward(Variable(torch.from_numpy(self.X)).float())
        accu = np.mean(np.round(predict.data.numpy()) == self.y)
        print('Accuracy:', accu)
        print('Paramters:')
        for param in model.parameters():
            print(param.data)
            print(torch.sum(param.data))

    def _write_model(self, param, coef, score, model_name):
        output = {}
        output['param'] = param
        output['coef'] = coef
        output['score'] = score
        output['time'] = str(datetime.datetime.now())
        output['name'] = model_name
        print(output)
        feed = pickle.load(open(self.model_log, 'rb'))
        with open(self.model_log, 'wb') as f:
            feed.append(output)
            pickle.dump(feed, f)


def get_test_data(download_path='tensor/data'):
    """Download and processing of test data."""
    DATA_PATH = os.path.abspath(download_path)
    TESTDATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    download_data(TESTDATA_URL, DATA_PATH)

    testdata = pd.read_csv(os.path.join(DATA_PATH, 'sonar.all-data'), header=None)
    print("Test data has ", testdata.shape[0], "rows")
    print("Test data has ", testdata.shape[1], "features")
    X = scale(testdata.iloc[:, :-1])

    y = testdata.iloc[:, -1].values
    encoder = LabelEncoder()
    encoder.fit(np.unique(y))
    y = encoder.transform(y)

    return X, y


if __name__ == '__main__':
    DATA_PATH = 'tensor/data'
    model_comparision_file = os.path.join(DATA_PATH, 'model.comparisions')
    X, y = get_test_data(DATA_PATH)

    pytorchmodel = pytoch_linear(X, y, model_comparision_file, True)
    pytorchmodel.run(penal='l0')
