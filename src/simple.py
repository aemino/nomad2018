import keras.backend as K
from keras.losses import logcosh
from keras.callbacks import Callback, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import preprocess_data, build_net

train, test, targets, transform_feats, feats, all_data, x_train, y_train, x_test = preprocess_data()

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
	
	def on_epoch_end(self, batch, logs={}):
		loss = logs.get('loss')
		if loss > 0.03:
			return

		self.losses.append(loss)

history = LossHistory()

layers = [128] * 8

model = build_net(layers=layers, input_dim=x_train.shape[1], output_dim=y_train.shape[1], lr=0.001, loss='mse')

early_stopping = EarlyStopping(monitor='loss', patience=100)
model.fit(x=x_train, y=y_train, epochs=2000, callbacks=[history, early_stopping])

y_predict = model.predict(x_train)

test_predict = np.expm1(model.predict(x_test))
columns = targets
predictions = pd.DataFrame({columns[i]: test_predict[:,i] for i in range(len(columns))})
predictions.insert(loc=0, column='id', value=list(range(1, len(test_predict) + 1)))
predictions.to_csv('submission.csv', index=False)
print(predictions.head())

# graph losses

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.title('loss')
plt.plot(history.losses)

#plt.subplot(2, 2, 3)
#plt.title(targets[0])
#plt.plot(y_train.values[:50,0] - y_predict[:50,0])

#plt.subplot(2, 2, 4)
#plt.title(targets[1])
#plt.plot(y_train.values[:50,1] - y_predict[:50,1])

plt.show()
