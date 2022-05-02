import numpy as np 
from common import *

means = np.array([0,0])
covariances = np.array([[3.0, 2.9], [2.9, 3.0]])

print("Generating 500 samples from 2d Gaussian distribution")
x, y = np.random.multivariate_normal(means, covariances, 500).T

plt.scatter(x, y)
plt.xlabel("z1")
plt.ylabel("z2")
plt.title("2-dimensional Gaussian samples")
plt.show()

print("============================================")
print("Using Gibbs sampling")
data = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
sample_mean  = np.mean(data, axis= 0)
sample_covar = np.mean(np.multiply(data[:, :, np.newaxis], data[:, np.newaxis, :]), axis= 0)

def Gibbs_Sampler(z1, z2):
	samples = np.zeros((100+1, 2))
	samples[0,:] = np.array([z1,z2])

	for i in range(100):
		sample_m = sample_mean[0] + (sample_covar[0, 1]*(z2 - sample_mean[1]) / sample_covar[1,1])
		sample_s = sample_covar[0,0] - (sample_covar[0,1]*sample_covar[0,1]/sample_covar[1,1])
		z1 = np.random.normal(sample_m, sample_s, 1)[0]

		sample_m = sample_mean[1] + (sample_covar[0, 1]*(z2 - sample_mean[0]) / sample_covar[0,0])
		sample_s = sample_covar[1,1] - (sample_covar[0,1]*sample_covar[0,1]/sample_covar[0,0])
		z2 = np.random.normal(sample_m, sample_s, 1)[0]

		samples[i+1,:] = np.array([z1, z2])

	return samples[:, 0], samples[:, 1]


z1 = -4.0 
z2 = -4.0 
sample_z1, sample_z2 = Gibbs_Sampler(z1, z2)

plt.scatter(x, y)
plt.plot(sample_z1, sample_z2, color="red")
plt.title("Gibbs Path on Distribution")
plt.show()

print("============================================")
print("Using HMC")

def HMC_Sampler(z1, z2):
	samples = np.zeros((100+1, 2))
	samples[0,:] = np.array([-4.0, -4.0])
	epsilon = 0.1
	leapfrog = 20
	z = np.array([z1, z2])
	mean = np.array([0,0])
	covar = np.array([[3.0, 2.9], [2.9, 3.0]])

	accepted_samples = 20

	for i in range(100):
		p = np.random.normal(0.0, 1.0, 2)

		pot = log_joint_gradient2(z, mean[np.newaxis, :], covar)
		pt = p.flatten()
		hamilton = (1e-10*0.5*np.sum(np.multiply(pt, pt))) + pot
		old_grad = log_joint_gradient2(z, mean[np.newaxis, :], covar)
		new_z = np.copy(z)
		new_grad = np.copy(old_grad)

		for j in range(leapfrog):
			p -= (epsilon / 2.0) * new_grad
			new_z += epsilon*p
			new_grad = log_joint_gradient2(new_z, mean[np.newaxis, :], covar)
			p -= (epsilon/2.0)*new_grad

		p = -p
		pot = log_joint_gradient2(new_z, mean[np.newaxis, :], covar)
		pt = p.flatten()
		hamilton_new = (1e-10*0.5*np.sum(np.multiply(pt, pt))) + pot

		if accept_log(np.exp(-hamilton), np.exp(-hamilton_new)):
			z = new_z
			accepted_samples += 1

		samples[i+1,:] = z

	return samples[:, 0], samples[:, 1]

z1 = -4.0
z2 = -4.0
sample_z1, sample_z2 = HMC_Sampler(z1, z2)

plt.scatter(x, y)
plt.plot(sample_z1, sample_z2, color="red")
plt.title("HMC Path on Distribution")
plt.show()
