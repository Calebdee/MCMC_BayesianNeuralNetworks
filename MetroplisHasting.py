import numpy as np
import matplotlib.pyplot as plt
from common import *

def metroplis_hastings(prior, initial, burn=True):
	accept = []
	sample = []
	final = []
	iters = 100000 if burn else 50000

	X = initial
	for i in range(iters):
		new = np.zeros(X.shape)
		new[0] = transition(X[0], X[1])
		new[1] = tau

		x_likelihood = log_joint(X[0])
		new_likelihood = log_joint(new[0])

		sample.append(new[0])
		if (to_accept(x_likelihood, new_likelihood)):
			X = new
			accept.append(new[0])
		if i % 10 == 0:
			final.append(X[0])

	if burn:
		return X
	else:
		return np.array(accept), np.array(final)


taus = [0.01, 0.1, 0.2, 0.5, 1.0]

print("Performing Metroplis-Hasting")
for tau in taus:
	burned_x = metroplis_hastings(prior=prior, initial=np.array([0, tau]), burn=True)
	accepted, finals = metroplis_hastings(prior=prior, initial=np.array([0, tau]), burn=False)

	print("Accepted for tau=" + str(tau) + " is " + str(accepted.shape[0]))

	z = np.arange(-3, 3+0.01, 0.01)
	N = gass_hermite_quad(affine_sigmoid, degree= 100, m= 10, c= 3)
	pz = np.multiply(np.exp(-np.multiply(z, z)), affine_sigmoid(z))/N

	plt.plot(z, pz, label="Ground-truth", color="black")
	plt.hist(finals, bins=50, density="True", alpha=0.5, label="Samples", color='firebrick')
	plt.ylim((-0.05, 1.0))
	plt.title("Metroplis-Hasting Tau="+str(tau))
	plt.legend(loc="upper left")
	plt.show()

