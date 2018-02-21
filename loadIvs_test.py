import os
import numpy as np
import re

class IvectorLoader(object):
	"""docstring for IvectorLoader"""
	def __init__(self, **kwargs):
		super(IvectorLoader, self).__init__()
		self.short_ivs_dir = kwargs.get("short_ivs_dir", "ivectors")
		self.long_ivs_dir = kwargs.get("long_ivs_dir", "long_ivectors")
		self.batch_size = kwargs.get("batch_size", 128)
		self.state = kwargs.get("task_type", "train")
		assert self.state in ["train", "test"]

		#storage: file id : [ short iv1, short iv2......]
		short_ivs_files = [one for one in os.listdir(self.short_ivs_dir) if one.endswith("ark") and self.state in one] 
		self.file2short_ivs = dict()
		for short_ivs_file in short_ivs_files:
			short_ivs_file = os.path.join(self.short_ivs_dir, short_ivs_file)
			with open(short_ivs_file) as source:
				for line in source:
					if "[" not in line or "]" not in line:
						break
					datas = line.split()
					name = "_".join(datas[0].split("_")[:-1])

					feats = [one for one in datas[1:] if "[" not in one and "]" not in one]
					feats = map(float, feats)
					if len(feats) != 400:
						break
					if name not in self.file2short_ivs:
						self.file2short_ivs[name] = [feats]
					else:
						self.file2short_ivs[name].append(feats)
		#storage: file id : long iv
		long_ivs_files = [one for one in os.listdir(self.long_ivs_dir) if one.endswith("ark") and self.state in one]
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