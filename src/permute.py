from keras.callbacks import EarlyStopping
import keras.backend as K
import numpy as np
import random as rn
from util import preprocess_data, build_net

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

num_perms = 50
num_runs = 50
num_epochs = 1000
gpus = 2

activations = ['tanh', 'relu', 'sigmoid']
num_layer_range = 64
epochs_range = 10
lr_range = 200

train, test, targets, transform_feats, feats, all_data, x_train, y_train, x_test = preprocess_data()

losses = []

exp_x = rn.sample(range(1, num_layer_range + 1), num_perms)

for p in range(num_perms):
	num_layers = exp_x[p]
	
	perm_losses = []
	for i in range(num_runs):
		print('Testing network permutation %s/%s run %s/%s' % (p + 1, num_perms, i + 1, num_runs))
		model = build_net(layers=[16] * num_layers, input_dim=x_train.shape[1], output_dim=y_train.shape[1], gpus=gpus)
		early_stopping = EarlyStopping(monitor='loss', patience=4)
		history = model.fit(x=x_train, y=y_train, epochs=num_epochs, callbacks=[early_stopping])
		perm_losses.append(history.history.get('loss')[-1])
		del model
		K.clear_session()
		print()
	
	perm_loss = sum(perm_losses) / float(num_runs)
	losses.append([num_layers, perm_loss])
	print('Average loss for network permutation %s: %s' % (p + 1, perm_loss))

loss_plot = np.array(losses)
# black magic code that sorts by the first column
loss_plot = loss_plot[loss_plot[:,0].argsort()]

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.title('num. layers')
plt.plot(loss_plot[:,0], loss_plot[:,1], label='loss')

plt.savefig('graph.png')
