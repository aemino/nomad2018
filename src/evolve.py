from keras.callbacks import EarlyStopping
import keras.backend as K
import numpy as np
import random as rn
from util import preprocess_data, build_net

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot

gpus = 2

population_size = 20
population_retain_fraction = 0.4
mutate_chance = 0.2
fitness_runs = 5

genetic_props = [
	# neurons
	(1, 1024),

	# layers
	(1, 256),

	# lr
	(0.001, 0.01)
]

train, test, targets, transform_feats, feats, all_data, x_train, y_train, x_test = preprocess_data()

class Genetics():
	def __init__(self, parents=None, props=None):
		if parents != None:
			props = self.combine(parents)

		if props == None:
			props = self.gen_rand_props()
		
		self.props = props
	
	def gen_rand_props(self):
		return [self.gen_rand_prop(prop) for prop in genetic_props]

	def gen_rand_prop(self, prop):
		if type(prop) is range:
			if type(*prop) is int:
				return rn.randint(*prop)
			else:
				return rn.uniform(*prop)
		if type(prop) is list:
			return rn.choice(prop)
	
	def combine(self, sources):
		return [rn.choice(props) for props in zip(*sources)]

	def mutate(self):
		if rn.random() < mutation_factor:
			new_props = list(self.props)
			mut_ind = rn.randint(0, len(new_props))
			new_props[mut_ind] = self.gen_rand_prop(genetic_props[mut_ind])
			return new_props

		return self.props

class Network():
	def __init__(self, props):
		self.props = props
		self.scores = []
	
	def eval_score(self):
		for i in range(fitness_runs):
			num_neurons, num_layers, lr = *self.props
			layers = [num_neurons] * num_layers
			model = build_net(layers=layers, input_dim=x_train.shape[1], output_dim=y_train.shape[1], lr=lr, gpus=gpus)
			early_stopping = EarlyStopping(monitor='loss', patience=4)
			history = model.fit(x=x_train, y=y_train, epochs=1000, callbacks=[early_stopping])

			self.scores.append(history.history.get('loss')[-1])

			del model
			K.clear_session()
		
		return sum(self.losses) / float(fitness_runs)

class Population():
	def __init__(self):
		self.population = [Genetics() for i in range(population_size)]
		self.generation = 0
	
	def evolve(self):
		print('Evolving generation %s' % (self.generation))
		networks = [Network(props) for props in self.population]

		scores = [[network.eval_score(), network] for network in networks]

		avg_score = sum(scores) / float(len(scores))
		print('Average score for generation %s: %s' % (self.generation, avg_score))

		npscores = np.array(scores)
		npscores = npscores[npscores[:,0].argsort()]

		num_retain = int(population_retain_fraction * population_size)
		new_population = scores[:,1][:num_retain].tolist()

		parents = rn.shuffle(list(new_population))[:2]

		[new_population.append(Genetics(parents=parents)) for i in range(population_size - len(new_population))]

		generation += 1

population = Population()
while True:
	population.evolve()

