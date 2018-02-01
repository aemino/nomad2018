import keras.backend as K
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np

from util import preprocess_data, build_net

train, test, targets, transform_feats, feats, all_data, x_train, y_train, x_test = preprocess_data()

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
	
	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

history = LossHistory()

activation = 'tanh'
layers = [16] * 128

model = build_net(activation=activation, layers=layers, input_dim=x_train.shape[1], output_dim=y_train.shape[1])
model.fit(x=x_train, y=y_train, epochs=5, callbacks=[history])

y_predict = model.predict(x_train)

# graph losses

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.title('loss')
plt.plot(history.losses)

plt.subplot(2, 2, 3)
plt.title(targets[0])
plt.plot(y_train.values[:50,0] - y_predict[:50,0])

plt.subplot(2, 2, 4)
plt.title(targets[1])
plt.plot(y_train.values[:50,1] - y_predict[:50,1])

plt.show()
