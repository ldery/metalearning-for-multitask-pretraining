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


from .superclasses import *
from .augmentations import augmentations_and_fns, is_augmentation, apply_augs, normalize_augmentation
import torch
import torchvision
import numpy as np
import math
from collections import defaultdict
import pdb


class CIFAR100:
	def __init__(
					self, **kwargs
				):
		# Get the cifar100 dataset from torchvision
		save_path = "~/" if 'save_path' not in kwargs else kwargs['save_path']
		tform = torchvision.transforms.Compose([
								torchvision.transforms.ToTensor(),
							])
		self.train = torchvision.datasets.CIFAR100(save_path, train=True, download=True, transform=tform)
		self.test = torchvision.datasets.CIFAR100(save_path, train=False, download=True, transform=tform)
		self.super_classlist = cifar100_super_classes
		self._group_data(flatten=kwargs['flatten'], prim_datafrac=kwargs['prim_datafrac'], prim_key=kwargs['prim_key'])
		self.aug_fns = augmentations_and_fns()

	# Map old classes to new classes under super-class.
	def _create_new_classes(self, orig_dict, reversed_name_dict, flatten):
		this_dict_ = defaultdict(list)
		for (x, y) in orig_dict:
			key_ = reversed_name_dict[y][0]  # Get the super-class for the class of this image
			new_class = reversed_name_dict[y][1] # Get the new id within this super-class
			if len(this_dict_[key_]) == 0:
				this_dict_[key_] = [[] for _ in range(CLASS_SIZES[key_])]
			this_dict_[key_][new_class].append(x.flatten() if flatten else x)
		return this_dict_

	def _group_data(self, flatten=False, prim_datafrac=1.0, prim_key='people'):
		# get the reversed dictionary. Gives mapping of current class to id within superclass
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
		
		# Need to down-size the dataset
		if prim_datafrac < 1.0:
			# Todo [ldery] - you should extend this for when the primary task is multiple tasks.
			# Code checked through pdb run.
			for set_ in [self.train_dict_, self.val_dict_]:
				prim_data = set_[prim_key]
				max_class_len = max([len(x) for x in prim_data])
				num_samples = int(prim_datafrac * max_class_len)
				assert num_samples > 0, 'Invalid number of samples requested'
				for idx in range(len(prim_data)):
					prim_data[idx] = prim_data[idx][:num_samples]
				set_[prim_key] = prim_data

	def _get_iterator(self, classes, batch_sz, split='train', shuffle=True):
		chosen_dict = self.train_dict_
		if split == 'val':
			chosen_dict = self.val_dict_
		elif split == 'test':
			chosen_dict = self.test_dict_
		else:
			assert "Invalid value of split given {}".format(split)

		# Calcuate the examples per-sub-clas. This equalizes the # of examples per-class in each batch.
		# Maybe might be better to not do this. Confirm - Todo [ldery]
		per_sub_class = math.ceil(batch_sz / (NUM_PER_SUPERCLASS * len(classes)))
		assert per_sub_class > 0, 'Samples per-sub-class must be > 0'

		# Setup the random idxs
		iter_idx_dict = defaultdict(list)
		max_iters = -1
		for class_ in classes:
			if is_augmentation(class_): # Will apply augmentation to the chosen members of the class
				continue
			class_name = class_
			data_ = chosen_dict[class_name]
			if shuffle:
				idxs = [np.random.permutation(len(x)) for x in data_]
			else:
				idxs = [np.arange(len(x)) for x in data_]
			iter_idx_dict[class_] = idxs
			if max_iters < 0:
				max_iters = min([len(x) for x in data_])
			else:
				max_iters = min(max_iters, *[len(x) for x in data_])
		# TODO [ldery] :
		# This approach means some of the data might not be looked at. Look into it.
		max_iters = int(max_iters / per_sub_class)

		assert max_iters > 0, "The maximum number of examples for a class must be > 0"
		num_iters = 0
		while True:
			batch_dict = {}
			num_iters += 1
			for class_ in classes:
				if is_augmentation(class_):
					continue
				class_name = class_
				data_ = chosen_dict[class_name]
				idxs = iter_idx_dict[class_]
				xs, ys = [], []
				# Todo [ldery] - write a test to make sure it's a different batch shown at every iteration
				# Tested [used main below]
				for id_, vals in enumerate(idxs):
					xs.extend([data_[id_][i] for i in vals[:per_sub_class]])
					ys.extend([id_ for _ in range(per_sub_class)])
					idxs[id_] = vals[per_sub_class:]
				apply_augs(class_, (xs, ys), batch_dict, classes, self.aug_fns)
				batch_dict[class_] = normalize_augmentation((xs, ys))

			yield batch_dict
			if num_iters == max_iters:
				break


def visualize(dict_, dict_name):
	if not os.path.exists(dict_name):
		os.makedirs(dict_name)
	for k, v in dict_.items():
		idxs = np.random.choice(50, size=NUM_PER_SUPERCLASS)
		chosen = []
		for id_, class_ in enumerate(v):
			chosen.append(class_[idxs[id_]])
		loc = "{}/{}".format(dict_name, k)
		if not os.path.exists(loc):
			os.makedirs(loc)

		for id_, img in enumerate(chosen):
			np_img = img.cpu().numpy()
			np_img = np.transpose(np_img, (1, 2, 0))
			plt.imshow(np_img)
			plt.savefig("{}/{}/{}.png".format(dict_name, k, id_))
			plt.show()
			plt.close()


if __name__ == '__main__':
	import os
	import matplotlib.pyplot as plt
	data = CIFAR100(flatten=False, prim_datafrac=0.1, prim_key='medium-sized_mammals')
	b1 = None
	all_b = []
	for batch in data._get_iterator(['medium-sized_mammals', 'small_mammals'], 16, split='val'):
		all_b.append(batch)
	pdb.set_trace()
	# visualize(data.train_dict_, 'train')
	# visualize(data.val_dict_, 'val')
	# visualize(data.test_dict_, 'test')
	# pdb.set_trace()

