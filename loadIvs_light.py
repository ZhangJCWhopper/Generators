import os
import numpy as np
import re
from subprocess import call

class IvectorLoader(object):
	"""
	docstring for IvectorLoader
	this is for load train male speakers' ivectors 
	I forget all
	Hahahahah
	file: utterance file
	
	"""
	def __init__(self, **kwargs):

		super(IvectorLoader, self).__init__()
		self.short_ivs_file = kwargs.get("short_ivs_file", "ivectors.ark")

		task = kwargs.get("task", "train")
		gender = kwargs.get("gender", "male")

		print("Attention: this is light version")
		print("Load short utt i-vectors from "+self.short_ivs_file)
		
		def filter(target, task=task, gender=gender):
			if gender == "male":
				ged = "_male."
			elif gender == "female":
				ged = "_female"
			else:
				ged = "."

			if target.endswith(".ark") and task in target and ged in target:
				return True
			return False

		short_ivs_files = [self.short_ivs_file] 
		self.carrier = []
		short_ivs_count = 0
		for short_ivs_file in short_ivs_files:
			with open(short_ivs_file) as source:
				for line in source:
					if "[" not in line or "]" not in line:
						break
					datas = line.split()
					name = datas[0]

					feats = map(float, [one for one in datas[1:] if "[" not in one and "]" not in one])

					if len(feats) not in [150, 400]:
						break
					self.carrier.append([name, feats])
					
					short_ivs_count += 1
		print("Load "+str(short_ivs_count)+" short utt i-vectors.")

	def get_eval_basis(self):
		#spk_id, picked short i-v, long i-v

		return self.carrier