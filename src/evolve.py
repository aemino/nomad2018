import gc
from keras.callbacks import EarlyStopping
import keras.backend as K
import numpy as np
import os
import pandas as pd
import random as rn
from util import preprocess_data, build_net, create_custom_session, ProgressBar

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = int(os.environ['NUM_GPUS'])

population_size = 50
population_retain_fraction = 0.4
population_stagnation_gens = 10
mutate_chance = 0.2
fitness_runs = 5

gene_schema = [
	# neurons
	(1, 256),

	# layers
	(1, 256),

	# lr
	(0.001, 0.01),

	# batch size exponent
	(2, 10)
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
		return [self.gen_rand_prop(schema) for schema in gene_schema]

	def gen_rand_prop(self, schema):
		if type(schema) is tuple:
			if type(schema[0]) is int:
				return rn.randint(*schema)
			else:
				return rn.uniform(*schema)
		if type(schema) is list:
			return rn.choice(schema)
	
	def combine(self, genes):
		return [rn.choice(props) for props in zip(*map(lambda g: g.props, genes))]

	def mutate(self):
		if rn.random() < mutation_factor:
			new_props = list(self.props)
			mut_ind = rn.randint(0, len(new_props))
			new_props[mut_ind] = self.gen_rand_prop(genetic_props[mut_ind])
			return new_props

		return self.props

class Network():
	def __init__(self, genes):
		self.genes = genes
	
	def eval_loss(self, progress):
		losses = []

		for i in range(fitness_runs):
			progress.display()

			with create_custom_session().as_default() as sess:
				num_neurons, num_layers, lr, batch_size_exp = self.genes.props
				layers = [num_neurons] * num_layers
				batch_size = 2 ** batch_size_exp
				model = build_net(
					layers=layers, input_dim=x_train.shape[1], output_dim=y_train.shape[1], lr=lr, gpus=gpus)
				early_stopping = EarlyStopping(monitor='loss', patience=4)
				history = model.fit(
					x=x_train, y=y_train, epochs=1000, batch_size=batch_size, callbacks=[early_stopping], verbose=0)

				losses.append(history.history.get('loss')[-1])

				sess.close()
				K.clear_session()

			progress.increment()
			progress.display()
		
		gc.collect()
		
		return sum(losses) / float(fitness_runs)

class Population():
	def __init__(self):
		self.population = [Genetics() for i in range(population_size)]
		self.generation = 0
		self.avg_losses = []
		self.lowest_avg_loss = None
		self.lowest_avg_loss_gen = 0
		self.lowest_avg_loss_data = None
		self.stagnant = False
	
	def evolve(self):
		print('Evolving generation %s' % (self.generation))
		networks = [Network(genes) for genes in self.population]

		progress = ProgressBar(population_size * fitness_runs)
		losses = [[network.eval_loss(progress), network] for network in networks]

		nplosses = np.array(losses)

		avg_loss = sum(nplosses[:,0].tolist()) / float(len(losses))
		self.avg_losses.append(avg_loss)
		
		if self.lowest_avg_loss == None or avg_loss < self.lowest_avg_loss:
			self.lowest_avg_loss = avg_loss
			self.lowest_avg_loss_gen = self.generation

			del self.lowest_avg_loss_data
			self.lowest_avg_loss_data = [[row[0], *(row[1].genes.props)] for row in losses]

		print('Average loss for generation %s: %s' % (self.generation, avg_loss))

		nplosses = nplosses[nplosses[:,0].argsort()]

		num_retain = int(population_retain_fraction * population_size)
		new_population = list(map(lambda n: n.genes, nplosses[:,1][:num_retain].tolist()))
		rn.shuffle(new_population)

		parents = new_population[:2]

		[new_population.append(Genetics(parents=parents)) for i in range(population_size - len(new_population))]

		self.population = new_population
		self.generation += 1

		self.update_graph()

		if self.generation - self.lowest_avg_loss >= population_stagnation_gens:
			self.stagnant = True

		gc.collect()
	
	def update_graph(self):
		plt.figure(figsize=(15, 10))
		plt.subplot(2, 1, 1)
		plt.title('num. generations vs avg. loss')
		plt.plot(self.avg_losses, label='loss')

		plt.savefig('graph.png')
		plt.close()

population = Population()
while not population.stagnant:
	population.evolve()

print('Population has reached an optimal state with an avg. loss of %0.5f' % (population.lowest_avg_loss))

pop_data = np.array(population.lowest_avg_loss_data)
columns = ['avg_loss', 'neurons', 'layers', 'lr', 'bse']
pop_df = pd.DataFrame({columns[i]: pop_data[:,i] for i in range(len(columns))})
pop_df.to_csv('lowest_avg_loss_pop.csv', index=False)

