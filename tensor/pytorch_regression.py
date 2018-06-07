"""Class and function for penalized regressions with tensorflow."""
import os
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
        :param loc_mean: mean of the normal of initial location parameters
        :param loc_sdev: standard deviation of initial location parameters
        :param beta: initial temperature parameter
        :param gamma: lower bound of "stretched" s
        :param zeta: upper bound of "stretched" s
        :param fix_temp: True if temperature is fixed
        """
        super(_L0Norm, self).__init__()
        self._origin = origin
        self._size = self._origin.weight.size()
        self.loc = nn.Parameter(torch.zeros(self._size).normal_(loc_mean,
                                                                loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(
            torch.zeros(1).fill_(beta))
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
        super(L12Linear, self).__init__(nn.Linear(in_features,
                                                  out_features,
                                                  bias=bias), **kwargs)
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


class pytorch_linear(object):
    """Penalized regresssion with L1/L2/L0 norm."""

    def __init__(self, X, y, model_log, overwrite=False, type='b'):
        """Penalized regression."""
        super(pytorch_linear, self).__init__()
        self.model_log = model_log
        self.overwrite = overwrite
        self.X = X
        self.y = y.reshape(len(y), 1)
        self.input_dim = X.shape[1]
        self.output_dim = 1
        self.type = type
        print(self.input_dim, self.output_dim)
        if not os.path.isfile(model_log):
            with open(model_log, 'w', encoding='utf-8') as f:
                pickle.dump([], f)
        elif overwrite:
            with open(model_log, 'wb') as f:
                pickle.dump([], f)
        assert os.path.isfile(self.model_log)

    def _model_builder(self, penal, **kwargs):
        if penal == 'l1':
            model = L12Linear(self.input_dim, self.output_dim, penal=penal)
        elif penal == 'l2':
            model = L12Linear(self.input_dim, self.output_dim, penal=penal)
        elif penal == 'l0':
            model = L0Linear(self.input_dim, self.output_dim, **kwargs)
        else:
            raise ValueError('incorrect norm specified')

        return model

    def _loss_function(self, labels, outputs):
        if self.type == 'c':
            loss = ((labels - outputs)**2).mean()
        elif self.type == 'b':
            loss = -(labels * torch.log(outputs)
                     + (1-labels)*torch.log(1-outputs)).mean()
        else:
            raise ValueError('wrong type specificed, either c or b')
        return loss

    def _accuracy(self, predict):
        if self.type == 'c':
            accu = np.corrcoef(self.y.flatten(),
                               predict.data.numpy().flatten())**2
            accu = accu[0][1]
        else:
            accu = np.mean(np.round(predict.data.numpy()) == self.y)
        return accu

    def run(self, penal='l1', lamb=0.01, epochs=201, l_rate=0.01, **kwargs):
        """Run regression with the given paramters."""
        model = self._model_builder(penal, **kwargs)

        optimizer = torch.optim.SGD(model.parameters(), lr=l_rate)
        save_loss = list()
        for _ in range(epochs):
            input = Variable(torch.from_numpy(self.X)).float()
            labels = Variable(torch.from_numpy(self.y)).float()
            optimizer.zero_grad()
            outputs, penalty = model.forward(input)
            loss = self._loss_function(labels, outputs)
            loss = loss + lamb*penalty
            loss.backward()
            optimizer.step()
            if (_+1) % 100 == 0:
                print("epoch {}, loss {}, norm {}".format(_, loss.item(),
                                                          penalty.item()))
                if np.allclose(save_loss[-1], loss.item(), 1e-3):
                    break
            save_loss.append(loss.item())
        predict, penal = model.forward(input)
        accu = self._accuracy(predict)
        print('Accuracy:', accu)
        print('Paramters:')
        coef = []
        for name, param in model.named_parameters():
            print(name)
            print(param.data)
            coef.append(param.data.numpy())
        param = {}
        param['lambda'] = lamb
        param['epoch'] = epochs
        param['penal'] = penal
        param['type'] = self.type
        self._write_model(param, coef, accu, 'torch_'+penal+'_'+self.type)

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
