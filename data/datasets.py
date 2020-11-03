#     Copyright 2020 Google LLC
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#         https://www.apache.org/licenses/LICENSE-2.0
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


import torch
from superclasses import *
import torchvision
import numpy as np
from collections import defaultdict


class CIFAR100:
	def __init__(
					self, **kwargs
				):
		# Get the cifar100 dataset from torchvision
		save_path = "~/" if 'save_path' not in kwargs else kwargs['save_path']
		normalize = torchvision.transforms.Normalize(
											mean=[0.4914, 0.4822, 0.4465],
											std=[0.2023, 0.1994, 0.2010]
										)
		tform = torchvision.transforms.Compose([
								torchvision.transforms.ToTensor(),
								normalize
							])
		self.train = torchvision.datasets.CIFAR100(save_path, train=True, download=True, transform=tform)
		self.test = torchvision.datasets.CIFAR100(save_path, train=False, download=True, transform=tform)
		self.super_classlist = cifar100_super_classes
		self._group_data(flatten=kwargs['flatten'])

	def _create_new_classes(self, orig_dict, reversed_name_dict, flatten):
		this_dict_ = defaultdict(list)
		for (x, y) in orig_dict:
			key_ = reversed_name_dict[y][0]
			new_class = reversed_name_dict[y][1]
			if len(this_dict_[key_]) == 0:
				this_dict_[key_] = [[] for _ in range(NUM_PER_CLASS)]
			this_dict_[key_][new_class].extend(x.flatten() if flatten else x)
		return this_dict_

	def _group_data(self, flatten=False):
		# get the reversed dictionary
		reversed_dict = {}
		for k, v in self.super_classlist.items():
			for c_id in v:
				reversed_dict[c_id] = (k, self.super_classlist[k].index(c_id))
		# Split the train into a train and test
		permutation = np.random.permutation(len(self.train))
		train_idxs = permutation[:int(0.8 * len(self.train))]
		train_list, val_list = [], []
		for idx_, pair in enumerate(self.train):
			if idx_ in train_idxs:
				train_list.append(pair)
			else:
				val_list.append(pair)
		self.train_dict_ = self._create_new_classes(train_list, reversed_dict, flatten)
		self.val_dict_ = self._create_new_classes(val_list, reversed_dict, flatten)
		self.test_dict_ = self._create_new_classes(self.test, reversed_dict, flatten)


if __name__ == '__main__':
	import pdb
	data = CIFAR100(flatten=False)
	pdb.set_trace()


