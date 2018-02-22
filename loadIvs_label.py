import os
import numpy as np
import tensorflow as tf
import re

class IvectorLoader(object):
	"""
	docstring for IvectorLoader
	I forget all
	Hahahahah
	file: utterance file

	"""
	def __init__(self, **kwargs):
		#super(IvectorLoader, self).__init__()
		self.short_ivs_dir = kwargs.get("short_ivs_dir", "ivectors")
		self.long_ivs_dir = kwargs.get("long_ivs_dir", "long_ivectors")
		self.batch_size = kwargs.get("batch_size", 128)
		self.ref_file = kwargs.get("ref", "spk2id")
		self.sre08_ref = kwargs.get("sre08_ref", "sre08")

		print ("Load short utt i-vectors from "+self.short_ivs_dir)
		print ("Load long utt i-vectors from " + self.long_ivs_dir)
		print ("Batch size " + str(self.batch_size))
		print ("This version has labels")
		self.spk2id = dict()
		#get spk id map table
		#uttname contains spk name, directly id
		with open(self.ref_file) as ref:
			for line in ref:
				conts = line.split()
				self.spk2id[conts[0]] = int(conts[1])
		#then, extra map for sre08
		#sre08: utt->spk->id
		self.sre08_utt2spk = dict()
		with open(self.sre08_ref) as sre08:
			for line in sre08:
		        contents = line.split()
		        f_name = contents[0]
		        spk = contents[1]
		        name = f_name.split('_')[-2] + "_" + f_name.split('_')[-1]
		        self.sre08_utt2spk[name] = spk

		#storage: file id : [ short iv1, short iv2......]
		short_ivs_files = [one for one in os.listdir(self.short_ivs_dir) if one.endswith("ark") and "female" not in one] 
		self.file2short_ivs = dict()
		short_ivs_count = 0
		for short_ivs_file in short_ivs_files:
			if "sre08" in short_ivs_file:
				sre08 = True
			else:
				sre08 = False
			short_ivs_file = os.path.join(self.short_ivs_dir, short_ivs_file)
			with open(short_ivs_file) as source:
				for line in source:
					if "[" not in line or "]" not in line:
						break
					datas = line.split()
					#delete the inner-utt id to get the long utt name
					name = "_".join(datas[0].split("_")[:-1])
					if sre08:
						name = "sre08" + name
					feats = [one for one in datas[1:] if "[" not in one and "]" not in one]
					feats = map(float, feats)
					if len(feats) != 400:
						break
					if name not in self.file2short_ivs:
						self.file2short_ivs[name] = [feats]
					else:
						self.file2short_ivs[name].append(feats)
					short_ivs_count += 1

		print "Load "+str(short_ivs_count)+" short utt i-vectors."
		#storage: file id : long iv
		long_ivs_count = 0
		long_ivs_files = [one for one in os.listdir(self.long_ivs_dir) if one.endswith("ark") and "female" not in one]
		self.file2long_ivs = dict()
		for long_ivs_file in long_ivs_files:
			sre08 = False
			if "sre08" in long_ivs_file:
				sre08 = True

			long_ivs_file = os.path.join(self.long_ivs_dir, long_ivs_file)
			with open(long_ivs_file) as source:
				for line in source:
					datas = line.split()
					name = datas[0]
					if sre08:
						name = "sre08" + name
					feats = [one for one in datas[1:] if '[' not in one and ']' not in one]
					feats = map(float, feats)
					if name in self.file2long_ivs:
						print "Fatal error", name, "duplicated"
					self.file2long_ivs[name] = feats
					long_ivs_count += 1
		print "Load "+str(long_ivs_count)+" long utt i-vectors."
		self.file_list = self.file2short_ivs.keys()
		self.current_file_index = 0;
		self.current_index = 0;
	
	def next_batch(self):
		batch = []
		current_file = self.file_list[self.current_file_index]

		while len(batch) <self.batch_size:

			if self.current_index > len(self.file2short_ivs[current_file]) - 1:
				#one long utterance is finished, convert to next utterance
				self.current_index = 0
				self.current_file_index += 1
				self.current_file_index %= len(self.file_list)
				current_file = self.file_list[self.current_file_index]
			batch.append((self.file2short_ivs[current_file][self.current_index], self.file2long_ivs[current_file]))

			self.current_index += 1

		return np.array(batch)

	def whole_set(self):
		short_utt_ivs = []
		long_utt_ivs = []
		for one in self.file_list:
			short_ivs_crt = self.file2short_ivs[one]
			short_utt_ivs += short_ivs_crt
			l = len(short_ivs_crt)
			long_utt_ivs += [self.file2long_ivs[one]] * l
		print "Whole set size " + str(len(short_utt_ivs))
		return np.array(short_utt_ivs), np.array(long_utt_ivs)

	def make_tfrecord(self, record_file="train.tfrecord"):
		#one utt, one file
		count = 0
		writer = tf.python_io.TFRecordWriter(record_file)
		for utt_name in self.file_list:
			short_ivs_list = self.file2short_ivs[utt_name]			
			long_iv = self.file2long_ivs[utt_name]

			for short_iv in short_ivs_list:
				example = tf.train.Example(
					features=tf.train.Features(
						feature={"short_ivector": tf.train.Feature(float_list=tf.train.Floatlist(value=short_iv))},
						feature={"long_ivector": tf.train.Feature(float_list=tf.train.Floatlist(value=long_iv))}
					)
				)
				serialized = example.SerializeToString()
				writer.write(serialized)
				count += 1
		writer.close()
		print "write done %d samples"%(count,)