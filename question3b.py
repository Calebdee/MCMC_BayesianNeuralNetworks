import numpy as np 
import pandas as pd
import scipy.stats as stats
from common import *

train = pd.read_csv("data/bank-note/train.csv").to_numpy()
test = pd.read_csv("data/bank-note/test.csv").to_numpy()
train_labels = train[:,-1]
test_labels = test[:,-1]

train_X = train[:,0:-1]
test_X = test[:,0:-1]

def gibbs(X, y, weights, burn=False):
	iters = 100000 if burn else 10000
	num = X.shape[1]
	wid = X.shape[0]
	a = np.eye(num) + np.sum(np.matmul(X[:, :, np.newaxis], X[:, np.newaxis, :]), axis= 0)
	weights_cov = np.linalg.inv(a)

	pz, z = sigmoid(weights, X)
	sigma = 1

	posteriors = []

	for i in range(iters):
		new_weights_mean = np.matmul(weights_cov, np.sum(z*X, axis=0)[:, np.newaxis])
		weights = np.random.multivariate_normal(new_weights_mean[:,0], weights_cov, 1).T
		pz, new_z = sigmoid(weights, X)

		low = np.zeros((wid, 1))
		upp = 1e8 * np.ones((wid, 1))
		low[y < 0.5, :] = -1e8
		upp[y < 0.5, :] = 0

		new_x = stats.truncnorm((low - new_z) / sigma, (upp - new_z) / sigma, loc=new_z)
		new_z = new_x.rvs((wid, 1))

		z = copy.deepcopy(new_z)

		if i % 10 == 0:
			posteriors.append(weights)

	if burn:
		return weights
	else:
		return posteriors

def posterior_prediction(test_data, test_label, posterior_samples):
    num = posterior_samples.shape[0]
    avg_pred_test = np.zeros((num,))
    avg_pred_log_lld = np.zeros((num,))
                    
    for k in range(num):
        w_sampled = posterior_samples[k]
        
        pred_test, _ = sigmoid(w_sampled, test_data)
        acc = accuracy(pred_test, test_label) 
        pred_likelihood = prediction_likelihood(test_data, test_label, w_sampled)
        avg_pred_test[k] = acc
        avg_pred_log_lld [k] = np.log(pred_likelihood)
        
    return np.mean(avg_pred_test[:num]), np.mean(avg_pred_log_lld[:num]) 

weights = np.zeros((train_X.shape[1]))
weights_burn = gibbs(train_X, train_labels, weights, burn=True)
finals = gibbs(train_X, train_labels, weights_burn)
print(posterior_prediction(test_data, test_labels, finals))