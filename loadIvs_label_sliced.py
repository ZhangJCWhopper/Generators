import os
import numpy as np
import re

class IvectorLoader(object):
	"""
	docstring for IvectorLoader
	I forget all
	Hahahahah
	file: utterance file
	if we want label, we can use male only
	yes, do not use sre08 test
	"""
	def __init__(self, **kwargs):
		super(IvectorLoader, self).__init__()
		self.short_ivs_dir = kwargs.get("short_ivs_dir", "ivectors")
		self.long_ivs_dir = kwargs.get("long_ivs_dir", "long_ivectors")
		self.batch_size = kwargs.get("batch_size", 128)
		
		print "Load short utt i-vectors from "+self.short_ivs_dir
		print "Load long utt i-vectors from " + self.long_ivs_dir
		print "Batch size " + str(self.batch_size)

		self.in_sre08 = dict()

		#storage: file id : [ short iv1, short iv2......]
		short_ivs_files = [one for one in os.listdir(self.short_ivs_dir) if one.endswith("ark") and "test" not in one and "female" not in one] 
		self.file2short_ivs = dict()
		short_ivs_count = 0
		for short_ivs_file in short_ivs_files:
			short_ivs_file = os.path.join(self.short_ivs_dir, short_ivs_file)
			with open(short_ivs_file) as source:
				for line in source:
					if "[" not in line or "]" not in line:
						break
					datas = line.split()
					name = "_".join(datas[0].split("_")[:-1])
					#if we want label, we should know whether it is from sre08
					if "sre08" in short_ivs_file:
						self.in_sre08[name] = True
					else:
						self.in_sre08[name] = False

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
		long_ivs_files = [one for one in os.listdir(self.long_ivs_dir) if one.endswith("ark") and "test" not in one and "female" not in one]
		self.file2long_ivs = dict()
		for long_ivs_file in long_ivs_files:
			long_ivs_file = os.path.join(self.long_ivs_dir, long_ivs_file)
			with open(long_ivs_file) as source:
				for line in source:
					datas = line.split()
					name = datas[0]

					feats = [one for one in datas[1:] if '[' not in one and ']' not in one]
					feats = map(float, feats)
					if name in self.file2long_ivs:
						print "Fatal error", name, "duplicated"
					self.file2long_ivs[name] = feats
					long_ivs_count += 1
		print("Load "+str(long_ivs_count)+" long utt i-vectors.")
		self.file_list = self.file2short_ivs.keys()
		self.current_file_index = 0
		self.current_index = 0
		self.whole_map = dict()
		self.sre08_lookup = dict()
	
	def get_label(self, map_file, sre08_file):
		#sre08: filename->spkid->id
		#other: filename->id
		self.whole_map = dict()
		with open(map_file) as src:
			for line in src:
				name, id = line.split()
				id = int(id)
				self.whole_map[name] = id
		#from file name to speaker id

		with open(sre08_file) as sre08:
			for line in sre08:
				name, spkid, _ = line.split()
				utt_name = name.split("_")[-2] + "_" + name.split("_")[-1]
				self.sre08_lookup[utt_name] = spkid
	
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

	def make_tfrecord(self, record_file="label.tfrecord"):
		#one utt, one file
		import tensorflow as tf
		count = 0

		file_counter = 0
		writer = tf.python_io.TFRecordWriter(record_file + str(file_counter))
		usedId = set()
		for utt_name in self.file_list:
			short_ivs_list = self.file2short_ivs[utt_name]			
			long_iv = self.file2long_ivs[utt_name]
			if self.in_sre08[utt_name]:
				#in sre08, do 2 steps
				utt_file = utt_name.split('-')[1]
				if utt_file not in self.sre08_lookup:
					continue
				mid_id = self.sre08_lookup[utt_file]
				id = self.whole_map[mid_id]
			else:
				#not in sre08
				#for sre04, xxx-sre04-test-xxxx, others xxx_xxx-xxx
				if "sre04" not in utt_name:
					spk_name = utt_name.split('-')[0]
				else:
					spk_name = '-'.join(utt_name.split('-')[:3])
				if spk_name in self.whole_map:
					id = self.whole_map[spk_name]
				else:
					continue
			usedId.add(id)

			for short_iv in short_ivs_list:
				if count > 0 and count % 9000 == 0:
					writer.close()
					file_counter += 1
					writer = tf.python_io.TFRecordWriter(record_file + str(file_counter))
					
				example = tf.train.Example(
					features=tf.train.Features(
						feature={"short_ivector": tf.train.Feature(float_list=tf.train.FloatList(value=short_iv)), 
								 "long_ivector": tf.train.Feature(float_list=tf.train.FloatList(value=long_iv)),
								 "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[id])),
								}
					)
				)
				serialized = example.SerializeToString()
				writer.write(serialized)
				count += 1

		writer.close()
		print "write done %d samples"%(count,)

		return list(usedId)