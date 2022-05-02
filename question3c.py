from bnn_functions import *

train = pd.read_csv("data/bank-note/train.csv")
test = pd.read_csv("data/bank-note/test.csv")
trainset = torchvision.data
train = torch.utils.data.DataLoader()
nodes = [10, 20, 50]
activations = ["relu", "tanh"]
lrs = [1e-3, 0.5e-3, 1e-4, 1e-5]
iters = 1000

for node in nodes:
	for activation in activations:
		for lr in lrs:
			model = BNN(node, activation)
			optimizer = torch.optim.Adam(model.parameters(), lr=lr)