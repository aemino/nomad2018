from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras import backend as K
import numpy as np
import pandas as pd
import os
import random as rn
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def seed_randomness(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	rn.seed(seed)

	session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

	tf.set_random_seed(seed)

	sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
	K.set_session(sess)

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

		model.add(Dense(layers.pop(0), input_dim=input_dim))

		for i, units in enumerate(layers):
			model.add(Dense(units, activation=activation))

		# output laye
		model.add(Dense(output_dim))

	optimizer = Adam(lr=lr)

	if gpus != None and gpus > 1:
		model = multi_gpu_model(model, gpus=gpus)
	
	model.compile(optimizer=optimizer, loss=loss)

	return model
