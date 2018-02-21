#!/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from numpy import linalg as LA
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import os
from subprocess import call
#some♂test♂kits♂for♂the♂networks


class PreparePLDA(object):
	"""docstring for PreparePLDA"""
	def __init__(self):
		#wtf
		self.noise_dim = 50
		self.iv_dim = 150
		self.noise_cov = 1.
		self.short_ivs_dir = "/work/jiacen/ivectors"
		self.short_ivs_file = "ivector.ark"
		self.long_ivs_dir = "/work/jiacen/long_ivectors"
		self.task = "test"
		self.turn = 1
		self.batch_size = 1
		self.eval_space = "/work/jiacen/eval"
		self.target_ark_file = os.path.join(self.eval_space, "generated_ivectors.ark")
		self.origin_ark_file = os.path.join(self.eval_space, "clipped_ivectors.ark")
		self.spk2utt_file = "/work/jiacen/spk2utt"

	def generate_samples(self, ckpt_dir):
		#ivector loader is full-funced, loads i-vectors from ark files under given dire
		import loadIvs_male
		checkpoint = tf.train.latest_checkpoint(ckpt_dir)
		reader = pywrap_tensorflow.NewCheckpointReader(checkpoint)
		network_vars = reader.get_variable_to_shape_map().keys()
		layer_num = 0
		for network_var in network_vars:
			if network_var.startswith("generator/gen"):
				num = int(network_var.split('/')[1].replace("gen", ""))
				layer_num = max(layer_num, num)
		layer_num += 1
		print("Layer number: %d" %(layer_num))
		def simple_generator(x, z):

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

		with tf.Session() as sess:
			with open(self.target_ark_file, "w") as out, open(self.origin_ark_file, "w") as origin_out, open(self.spk2utt_file, "w") as spk2utt_out:
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
					#write spk2utt
					str_buff = "%s %s\n" %(name, name)
					spk2utt_out.write(str_buff)

					cursor += 1
		
		LN_ark_file = self.target_ark_file.replace(".ark", "_LN.ark")
		LN_scp_file = self.target_ark_file.replace(".ark", "_LN.scp")
		LN_commands = ["ivector-normalize-length", "ark:"+self.target_ark_file, "ark,scp:%s,%s"%(LN_ark_file, LN_scp_file)]
		call(LN_commands)
		

		print "Finished!"

		"""
		copy-vector ark:%s ark,scp:%s,%s

		trials=data/sre08_trials/short2-short3-male.trials
		cat exp/ivectors_sre08_train_short2_male/spk_ivector.scp exp/ivectors_sre08_test_short3_male/ivector.scp > male.scp
		ivector-plda-scoring --num-utts=ark:exp/ivectors_sre08_train_short2_male/num_utts.ark \
		   "ivector-adapt-plda $adapt_opts exp/ivectors_train_male/plda scp:male.scp -|" \
		   scp:exp/ivectors_sre08_train_short2_male/spk_ivector.scp \
		   scp:exp/ivectors_sre08_test_short3_male/ivector.scp \
		   "cat '$trials' | awk '{print \$1, \$2}' |" foo; local/score_sre08.sh $trials foo
		
		cat exp/ivectors_sre08_train_short2_male/spk_ivector.scp ci.scp > ci_male.scp
		ivector-plda-scoring --num-utts=ark:exp/ivectors_sre08_train_short2_male/num_utts.ark "ivector-adapt-plda exp/ivectors_train_male/plda scp:ci_male.scp -|" scp:exp/ivectors_sre08_train_short2_male/spk_ivector.scp scp:ci.scp "cat '$trials' | awk '{print \$1, \$2}' |" ci_foo
		
		cat exp/ivectors_sre08_train_short2_male/spk_ivector.scp gi.scp > gi_male.scp
		ivector-plda-scoring --num-utts=ark:exp/ivectors_sre08_train_short2_male/num_utts.ark "ivector-adapt-plda exp/ivectors_train_male/plda scp:gi_male.scp -|" scp:exp/ivectors_sre08_train_short2_male/spk_ivector.scp scp:gi.scp "cat '$trials' | awk '{print \$1, \$2}' |" gi_foo
		"""

	def generate_samples_light(self, ckpt_dir):
		#i-vector loader is a light version, load i-vectors from the given ark file
		import loadIvs_light
		checkpoint = tf.train.latest_checkpoint(ckpt_dir)
		reader = pywrap_tensorflow.NewCheckpointReader(checkpoint)
		network_vars = reader.get_variable_to_shape_map().keys()
		layer_num = 0
		for network_var in network_vars:
			if network_var.startswith("generator/gen"):
				num = int(network_var.split('/')[1].replace("gen", ""))
				layer_num = max(layer_num, num)
		layer_num += 1

		def simple_generator(x, z):

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
		
		loader = loadIvs_light.IvectorLoader(short_ivs_file=self.short_ivs_file, task="test")
		raw_data = loader.get_eval_basis()
		
		short_ivs = [one[1] for one in raw_data]
		spks = [one[0] for one in raw_data]
		
		cursor = 0

		with tf.Session() as sess:
			with open(self.target_ark_file, "w") as out, open(self.origin_ark_file, "w") as origin_out, open(self.spk2utt_file, "w") as spk2utt_out:
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
							#generate turn times
							temp += existed_res[i][j]

						temp /= (self.turn * 1.0)
						final.append(np.float32(temp))
					#write generated
					str_buff = name + "  [ "
					for d in final:
						str_buff += (str(d) + " ")
					str_buff += "]\n"
					out.write(str_buff)
					
					cursor += 1
				
		LN_ark_file = self.target_ark_file.replace(".ark", "_LN.ark")
		LN_scp_file = self.target_ark_file.replace(".ark", "_LN.scp")
		LN_commands = ["ivector-normalize-length", "ark:"+self.target_ark_file, "ark,scp:%s,%s"%(LN_ark_file, LN_scp_file)]

		call(LN_commands)
		print "Finished!"
