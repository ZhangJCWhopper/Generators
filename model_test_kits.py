#!/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from numpy import linalg as LA
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import loadIvs_male
from subprocess import call
#some♂test♂kits♂for♂the♂networks


class PreparePLDA(object):
	"""docstring for PreparePLDA"""
	def __init__(self):
		#wtf
		self.noise_dim = 100
		self.iv_dim = 400
		self.noise_cov = .1
		self.short_ivs_dir = "~/workspace/ivectors"
		self.long_ivs_dir = "~/workspace/long_ivectors"
		self.task = "test"
		self.turn = 1
		self.batch_size = 1
		self.target_ark_file = "~/workspace/generated_ivectors.ark"
		self.origin_ark_file = "~/workspace/clipped_ivectors.ark"

	def generate_samples(self, ckpt_dir):
		checkpoint = tf.train.latest_checkpoint(ckpt_dir)
		reader = pywrap_tensorflow.NewCheckpointReader(checkpoint)
		#var_to_shape_map = reader.get_variable_to_shape_map()

		def simple_generator(x, z, layer_num=4):

			layers = [tf.concat([x, z], 1)]
			for i in range(layer_num):
				k = tf.Variable(reader.get_tensor("generator/gen%d/kernel" % i))
				b = tf.Variable(reader.get_tensor("generator/gen%d/bias" % i))
				layer = tf.matmul(layers[-1], k) + b
				layers.append(layer)
			return layers[-1]

		short_iv_holder = tf.placeholder(tf.float32, shape=[None, self.iv_dim], name="short_iv_holder")
		noise_holder = tf.placeholder(tf.float32, shape=[None, self.noise_dim], name="noise_holder")
		gened_ivs = simple_generator(short_iv_holder, noise_holder)
		
		loader = loadIvs_male.IvectorLoader(short_ivs_dir=self.short_ivs_dir, long_ivs_dir=self.long_ivs_dir, task=self.task)
		raw_data = loader.get_eval_basis()
		spks = [one[0] for one in raw_data]
		short_ivs = [one[1] for one in raw_data]

		cursor = 0

		with tf.Session() as sess, open(self.target_ark_file, "w") as out, open(self.origin_ark_file, "w") as origin_out:
			sess.run(tf.global_variables_initializer())
			while cursor < len(spks):
				#read data
				batch = short_ivs[cursor: cursor+self.batch_size]
				name = spks[cursor]
				#generate
				existed_res = []
				for i in range(self.turn):
					z = np.random.normal(.0, self.noise_cov, size=[self.batch_size, self.noise_dim])
					res = sess.run(gened_ivs, feed_dict={short_iv_holder: batch, noise_holder: z})
					existed_res.append(res[0])
				#compute average
				final = []
				for j in range(self.iv_dim):
					temp = .0
					for i in range(self.turn):
						temp += existed_res[i][j]

					temp /= 3.0
					final.append(np.float32(temp))
				#write generated
				str_buff = name + "  [ "
				for d in final:
					str_buff += (str(d) + " ")
				str_buff += "]\n"
				out.write(str_buff)
				#write original
				str_buff = name + "  [ "
				for d in batch[0]:
					str_buff += (str(d) + " ")
				str_buff += "]\n"
				origin_out.write(str_buff)

				cursor += 1

	def kaldi_preprocess(self):
		command = []
		#copy-feats

		#i-vector length normalization and get num-utts

		#adpat train plda model

		#compute plad scores

		#compute EER
