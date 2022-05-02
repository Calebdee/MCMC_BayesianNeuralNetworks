from common import *
import pandas as pd


def hybrid_mc(X, y, initial, epsilon, leapfrog, burn=False):
    dim = X.shape[1]
    weights = initial

    accepted = []
    sampled = []
    finals = []
    
    iters = 100000 if burn else 10000

    for i in range(iters):
        pred, dot_product = sigmoid(weights, X)
        old_grad = np.matmul(X.T, pred - y[:, np.newaxis])

        new_weights = np.copy(weights)             
        new_grad  = np.copy(old_grad) 

        p = np.random.normal(0.0, 1.0, (dim, 1))

        # Compute total energy (Hamiltonian)
        H = -Uzt(X, pred, y[:, np.newaxis], dot_product, weights) + Krt(p)

        # Leapfrog step, computing new gradient
        for j in range(leapfrog):  
            p -= (epsilon / 2.0) * new_grad
            new_weights += epsilon * p
            pred, dot_product = sigmoid(new_weights, X)
            new_grad = np.matmul(X.T, pred - y[:, np.newaxis])
            p -= (epsilon / 2.0) * new_grad

        # Compute new Hamiltonian
        pred, dot_product = sigmoid(new_weights, X)
        new_H  = -Uzt(X, pred, y[:, np.newaxis], dot_product, weights) + Krt(p) 
        
        sampled.append(new_weights)
        
        if accept_log(np.exp(-H), np.exp(-new_H)):            
            weights = new_weights
            accepted.append(new_weights)

        if i % 10 == 0:
        	finals.append(weights)

    if burn:
       	return weights 
    else:
        return accepted, sampled, finals
    


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


epsilons = [0.005, 0.01, 0.02, 0.05]
Ls = [10,20,50]

for epsilon in epsilons:
	for leapfrog in Ls:
		weights = np.zeros((train_data.shape[1], 1))
		weights_burn = hybrid_mc(train_data, train_labels, weights, epsilon, leapfrog, burn=True)
		accepted, sampled, finals = hybrid_mc(train_data, train_labels, weights_burn, epsilon, leapfrog)
		pred, predll = posterior_prediction(test_data, test_labels, np.array(finals))
		print("| e={ep:.3f} | L={leap:.3f} | accept rate {rate:.3f} | pred acc {pred:.3f} | log likelihood {predll:.3f} |"
			.format(ep=epsilon, leap=leapfrog, rate= len(accepted)/len(sampled), pred=pred, predll=predll))