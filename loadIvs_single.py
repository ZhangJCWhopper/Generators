import os
import numpy as np
import re
import tensorflow as tf

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
		self.ivs_dir = kwargs.get("ivs_dir", "ivectors")
		self.record_file = kwargs.get("tfrecord", "short.tfrecord")

		print "Load i-vectors from "+self.ivs_dir
		writer = tf.python_io.TFRecordWriter(self.record_file)
		#storage: file id : [ short iv1, short iv2......]
		ivs_files = [one for one in os.listdir(self.ivs_dir) if one.endswith("ark") and "test" not in one and "female" not in one] 
		self.file2ivs = dict()
		ivs_count = 0
		for ivs_file in ivs_files:
			ivs_file = os.path.join(self.ivs_dir, ivs_file)
			with open(ivs_file) as source:
				for line in source:
					if "[" not in line or "]" not in line:
						break
					datas = line.split()
					#name = "_".join(datas[0].split("_")[:-1])

					feats = [one for one in datas[1:] if "[" not in one and "]" not in one]
					feats = map(float, feats)
					if len(feats) != 400:
						break
					example = tf.train.Example(features=tf.train.Features(
						feature={"ivector": tf.train.Feature(float_list=tf.train.FloatList(value=feats)),}))
					serialized = example.SerializeToString()
					writer.write(serialized)
					ivs_count += 1
		writer.close()
		print "Wrote "+str(ivs_count)+" i-vectors."