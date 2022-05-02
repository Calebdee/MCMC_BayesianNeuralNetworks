import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt
import math

class LoadDataset(Dataset):
    def __init__(self, csv_path):

        self.all_data = pd.read_csv(csv_path).to_numpy()

        self.label_arr = self.all_data[:,-1]
        self.feature_arr  = np.hstack((self.all_data[:,0:-1], np.ones((self.all_data.shape[0],1))))
        self.data_len  = self.all_data.shape[0]


    def __getitem__(self, index):
        feature_as_tensor = torch.tensor(self.feature_arr[index]).type(torch.FloatTensor)
        label_as_tensor   = torch.tensor(self.label_arr[index])  .type(torch.LongTensor)
        return (feature_as_tensor, label_as_tensor)

    def __len__(self):
        return self.data_len


class BNN(nn.Module):
    def __init__(self, nodes, activation):
        super(BNN, self).__init__()
        self.nodes = nodes

        self.layer1 = BayesianLinear(INPUT_DIM     , self.nodes)
        self.layer2 = BayesianLinear(self.nodes, self.nodes)
        self.layer3 = BayesianLinear(self.nodes, NUM_CLASSES)

        self.activation = nn.ReLu() if activation=="relu" else nn.Tanh()

    def forward(self, x, sample= False):
        x = self.layer1(x, sample= sample)
        x = self.activation(x)
        x = self.layer2(x, sample= sample)
        x = self.activation(x)
        x = self.layer3(x, sample= sample)
        x = F.log_softmax  (x, dim=1)
        return x

    def sample_elbo(self, input, target, samples= 100, batch_size= 1000):
        outputs = torch.zeros(samples, batch_size, NUM_CLASSES).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        preds = torch.zeros(samples, batch_size).type(torch.LongTensor).to(DEVICE)
        corrects = torch.zeros(samples).to(DEVICE)

        for i in range(samples):
            outputs[i] = self.forward(input, sample= True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
            preds[i] = outputs[i].argmax(dim= 1)
            corrects[i] = preds[i].eq(target.view_as(preds[i])).sum()

        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(dim= 0), target, reduction='mean')
        loss = negative_log_likelihood
        correct = corrects.mean()

        return loss, negative_log_likelihood, log_prior, log_variational_posterior, correct

    def log_prior(self):
        return self.layer1.log_prior + self.layer2.log_prior + self.layer3.log_prior

    def log_variational_posterior(self):
        return self.layer1.log_variational_posterior + self.layer2.log_variational_posterior + self.layer2.log_variational_posterior

    