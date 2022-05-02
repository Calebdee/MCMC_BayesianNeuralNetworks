from common import *

def hybrid_mc(initial, epsilon, leapfrog, burn):
	accepted = []
	samples = []
	finals = []

	iters = 100000 if burn else 50000
	X = initial

	for i in range(iters):
		old = -log_joint(X)
		old_grad = -log_joint_gradient(X)

		new_x = copy.copy(X)
		new_grad = copy.copy(old_grad)

		p = np.random.normal(0.0, 1.0, 1)

		p_t = p.flatten()
		H = (0.5*np.sum(np.multiply(p_t, p_t))) + old

		for j in range(leapfrog):
			p -= epsilon*new_grad/2.0
			new_x += epsilon*p 
			new_grad = -log_joint_gradient(new_x)
			p -= epsilon*new_grad/2.0 

		p = -p
		new = -log_joint_gradient(new_x)
		p_t = p.flatten()
		new_H = (0.5*np.sum(np.multiply(p_t, p_t))) + new
		deltaH = new_H - H

		if (accept_hybrid(deltaH)):
			X = new_x
			accepted.append(new_x)

		if i % 10 == 0:
			finals.append(X)

	if burn:
		return X
	else:
		return np.array(accepted), np.array(finals)


epsilons = [0.005, 0.01, 0.1, 0.2, 0.5]
for epsilon in epsilons:
	burned_x = hybrid_mc(initial=np.array([0.0]), epsilon=epsilon, leapfrog=10, burn=True)
	accepted, finals = hybrid_mc(initial=np.array([0.0]), epsilon=epsilon, leapfrog=10, burn=False)

	print("Accepted for epsilon=" + str(epsilon) + " is " + str(accepted.shape[0]))

	plt.plot(z, pz, label="Ground-truth", color="black")
	plt.hist(finals, bins=50, density="True", alpha=0.5, label="Samples", color='firebrick')
	plt.ylim((-0.05, 1.0))
	plt.title("Hybrid MC with Leapfrog e="+str(epsilon))
	plt.legend(loc="upper left")
	plt.show()