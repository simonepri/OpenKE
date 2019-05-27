#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

def tf_resize(tensor, axis, size):
	shape = tensor.get_shape().as_list()
	osize = shape[axis]
	if osize == size:
		return tensor
	if (osize > size):
		shape[axis] = size
		return tf.slice(tensor, begin = (0,) * len(shape), size = shape)
	paddings = [[0, 0] for i in range(len(shape))]
	paddings[axis][1] = size - osize
	return tf.pad(tensor, paddings = paddings)

class TransD(Model):
	r'''
	TransD constructs a dynamic mapping matrix for each entity-relation pair by considering the diversity of entities and relations simultaneously.
	Compared with TransR/CTransR, TransD has fewer parameters and has no matrix vector multiplication.
	'''

	def _transfer(self, ent_e, ent_p, rel_p):
		rel_size = rel_p.get_shape()[-1]
		return rel_p * tf.reduce_sum(ent_e * ent_p, axis = -1, keepdims = True) + tf_resize(ent_e, axis = -1, size = rel_size);

	def _calc(self, ent1_t, ent2_t, rel_e, norm):
		if norm == (1, 1):
			return -tf.reduce_sum(tf.abs(ent1_t + rel_e - ent2_t), axis = -1, keepdims = True)
		elif norm == (2, 2):
			return -tf.reduce_sum(tf.square(ent1_t + rel_e - ent2_t), axis = -1, keepdims = True)
		elif norm[0] == norm[1]:
			return -tf.reduce_sum(tf.pow(ent1_t + rel_e - ent2_t, norm[0]), axis = -1, keepdims = True)
		return -tf.norm(tf.pow(ent1_t + rel_e - ent2_t, norm[0]), ord = norm[1], axis = -1, keepdims = True)

	def embedding_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#Defining required parameters of the model, including embeddings of entities and relations, entity transfer vectors, and relation transfer vectors
		self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [config.entTotal, config.ent_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [config.relTotal, config.rel_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.ent_transfer = tf.get_variable(name = "ent_transfer", shape = [config.entTotal, config.ent_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.rel_transfer = tf.get_variable(name = "rel_transfer", shape = [config.relTotal, config.rel_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
								"rel_embeddings":self.rel_embeddings, \
								"ent_transfer":self.ent_transfer, \
								"rel_transfer":self.rel_transfer}

	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#Get positive triples and negative triples for training
		#The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
		#The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
		pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True)
		neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True)
		#Embedding entities and relations of positive and negative triples
		#The shapes of pos_h_e, pos_t_e are (batch_size, 1, ent_size)
		#The shape of pos_r_e is (batch_size, 1, rel_size)
		#The shapes of neg_h_e, neg_t_e are (batch_size, negative_ent + negative_rel, ent_size)
		#The shape of neg_r_e is (batch_size, negative_ent + negative_rel, rel_size)
		pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, pos_h)
		pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, pos_t)
		pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, pos_r)
		neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, neg_h)
		neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, neg_t)
		neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, neg_r)
		#Getting the required parameters to transfer entity embeddings
		#The shapes of pos_h_p, pos_t_p are (batch_size, 1, ent_size)
		#The shape of pos_r_p is (batch_size, 1, rel_size)
		#The shapes of neg_h_p, neg_t_p are (batch_size, negative_ent + negative_rel, ent_size)
		#The shape of neg_r_p is (batch_size, negative_ent + negative_rel, rel_size)
		pos_h_p = tf.nn.embedding_lookup(self.ent_transfer, pos_h)
		pos_t_p = tf.nn.embedding_lookup(self.ent_transfer, pos_t)
		pos_r_p = tf.nn.embedding_lookup(self.rel_transfer, pos_r)
		neg_h_p = tf.nn.embedding_lookup(self.ent_transfer, neg_h)
		neg_t_p = tf.nn.embedding_lookup(self.ent_transfer, neg_t)
		neg_r_p = tf.nn.embedding_lookup(self.rel_transfer, neg_r)
		#Transfering entity embeddings into the relation space.
		#The shapes of pos_h_t, pos_t_t are (batch_size, 1, rel_size)
		#The shapes of neg_h_t, neg_t_t are (batch_size, negative_ent + negative_rel, rel_size)
		pos_h_t = self._transfer(pos_h_e, pos_h_p, pos_r_p)
		pos_t_t = self._transfer(pos_t_e, pos_t_p, pos_r_p)
		neg_h_t = self._transfer(neg_h_e, neg_h_p, neg_r_p)
		neg_t_t = self._transfer(neg_t_e, neg_t_p, neg_r_p)
		#The shape of pos_score is (batch_size, 1, 1)
		#The shape of neg_score is (batch_size, negative_ent + negative_rel, 1)
		pos_score = self._calc(pos_h_t, pos_t_t, pos_r_e, config.pnorm)
		neg_score = self._calc(neg_h_t, neg_t_t, neg_r_e, config.pnorm)
		#Calculating loss to get what the framework will optimize
		pairs_score = tf.maximum(pos_score - neg_score + config.margin, 0)
		self.loss = tf.reduce_sum(pairs_score)

	def predict_def(self):
		config = self.get_config()
		pre_h, pre_t, pre_r = self.get_predict_instance()
		pre_h_e = tf.nn.embedding_lookup(self.ent_embeddings, pre_h)
		pre_t_e = tf.nn.embedding_lookup(self.ent_embeddings, pre_t)
		pre_r_e = tf.nn.embedding_lookup(self.rel_embeddings, pre_r)
		pre_h_p = tf.nn.embedding_lookup(self.ent_transfer, pre_h)
		pre_t_p = tf.nn.embedding_lookup(self.ent_transfer, pre_t)
		pre_r_p = tf.nn.embedding_lookup(self.rel_transfer, pre_r)
		#Transfering entity embeddings into the relation space.
		#The shapes of pre_h_t, pre_t_t are (?, rel_size)
		pre_h_t = self._transfer(pre_h_e, pre_h_p, pre_r_p)
		pre_t_t = self._transfer(pre_t_e, pre_t_p, pre_r_p)
		#The shape of pre_score is (?, 1)
		pre_score = self._calc(pre_h_t, pre_t_t, pre_r_e, config.pnorm)
		self.predict = pre_score
