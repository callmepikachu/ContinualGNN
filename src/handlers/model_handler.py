import os
import torch

class ModelHandler(object):

	def __init__(self, save_path):
		super(ModelHandler, self).__init__()
		self.save_path = save_path
		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)

	def save(self, model, name):
		torch.save(model, os.path.join(self.save_path, name))

	def load(self, name):
		return torch.load(os.path.join(self.save_path, name))

	def not_exist(self):
		return not os.path.exists(self.save_path)