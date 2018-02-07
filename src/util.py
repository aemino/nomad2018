from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.backend import tensorflow_backend as T
import numpy as np
import pandas as pd
import os
import random as rn
from sklearn.preprocessing import MinMaxScaler
import sys
import tensorflow as tf

def create_custom_session():
	session_conf = tf.ConfigProto(allow_soft_placement=True)
	session_conf.gpu_options.per_process_gpu_memory_fraction = 0.5
	session_conf.gpu_options.allow_growth = True

	sess = tf.Session(config=session_conf)
	K.set_session(sess)
	return sess

def preprocess_data():
	train = pd.read_csv('data/train.csv')
	test = pd.read_csv('data/test.csv')

	targets = ['formation_energy_ev_natom', 'bandgap_energy_ev']
	transform_feats = ['spacegroup', 'number_of_total_atoms', 'percent_atom_al',
		'percent_atom_ga', 'percent_atom_in', 'lattice_vector_1_ang',
		'lattice_vector_2_ang', 'lattice_vector_3_ang', 'lattice_angle_alpha_degree',
		'lattice_angle_beta_degree', 'lattice_angle_gamma_degree']
	feats = transform_feats

	all_data = pd.concat([train[transform_feats], test])

	scaler = MinMaxScaler()
	scaler.fit(all_data[transform_feats])

	train[transform_feats] = scaler.transform(train[transform_feats])
	test[transform_feats] = scaler.transform(test[transform_feats])

	x_train = train.drop(['id'] + targets, axis=1)
	y_train = np.log1p(train[targets])

	x_test = test.drop(['id'], axis=1)

	return train, test, targets, transform_feats, feats, all_data, x_train, y_train, x_test

def build_net(activation='tanh', loss='msle', layers=[16] * 128, lr=0.002, input_dim=None, output_dim=None, gpus=None):
	with tf.device('/cpu:0'):
		model = Sequential()

		model.add(Dense(layers.pop(0), activation=activation, input_dim=input_dim))

		for i, units in enumerate(layers):
			model.add(Dense(units, activation=activation))

		# output laye
		model.add(Dense(output_dim))

	optimizer = Adam(lr=lr)

	if gpus != None and gpus > 1:
		model = multi_gpu_model(model, gpus=gpus)
	
	model.compile(optimizer=optimizer, loss=loss)

	return model

def display_progress(progress, total, bar_length=50, final=False):
	percent = (progress / total)
	fill_length = int(percent * bar_length)
	bar = '=' * fill_length
	arrow = '>' if fill_length < bar_length else ''
	bar_remain = '.' * (bar_length - fill_length)
	end = '\n' if final else '\r'

	sys.stdout.write(
		'[%s%s%s] - %0.2f%s [%s/%s]%s' % (bar, arrow, bar_remain, percent * 100, '%', progress, total, end))
	sys.stdout.flush()

class ProgressBar():
	def __init__(self, total, bar_length=50):
		self.total = total
		self.bar_length = bar_length
		self.progress = 0
	
	def increment(self):
		self.progress += 1
	
	def display(self):
		display_progress(self.progress, self.total, final=(self.progress == self.total))
