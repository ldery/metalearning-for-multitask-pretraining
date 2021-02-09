import torch
from torchvision import transforms
import numpy as np
import torchvision.transforms.functional as TF
import pdb

LIST_OF_AUGS = [
	'rand',
	'noise',
	'horzflip',
	'crop',
	'rotation'
]

NORM_FACTOR = 255.0
to_pil_tform = transforms.ToPILImage()
normalize_tform = transforms.Normalize(
											mean=[0.4914, 0.4822, 0.4465],
											std=[0.2023, 0.1994, 0.2010]
										)

def normalize_augmentation(data):
	xs, ys = data
	new_xs = [normalize_tform(x) for x in xs]
	new_xs = torch.stack(new_xs)
	return (new_xs.cuda(), torch.tensor(ys).cuda())

def rand_permutation(data):
	xs, ys = data
	rand_perm = np.random.permutation(len(ys))
	return normalize_augmentation((xs, ys[rand_perm]))

def noise_augmentation(data):
	xs, ys = normalize_augmentation(data)
	noisy_xs = torch.randn_like(xs)
	return (noisy_xs, ys)

def horzflip_augmentation(data):
	xs, ys = data
	flipped_xs = [transforms.RandomHorizontalFlip(p=1.0)(x) for x in xs]
	return normalize_augmentation((flipped_xs, ys))

def crop_augmentation(data):
	xs, ys = data
	resized_crop = transforms.RandomResizedCrop((32, 32), scale=(0.5, 1.0), ratio=(0.75, 1.33333), interpolation=2)
	results = []
	for idx in range(len(xs)):
		this_xs = xs[idx]
		pil_img = to_pil_tform(this_xs).convert("RGB")
		pil_img = resized_crop(pil_img)
		results.append(TF.pil_to_tensor(pil_img).float() / NORM_FACTOR)
	new_data = normalize_augmentation((results, ys))
	return new_data

def rotation_augmentation(data):
	xs, _ = data
	rotations = [0, 90, 180, 270]
	chosen_rots = np.random.choice(len(rotations), size=len(xs))
	results = []
	for idx, rot in enumerate(chosen_rots):
		this_xs = xs[idx]
		pil_img = to_pil_tform(this_xs).convert("RGB")
		pil_img = TF.rotate(pil_img, rotations[rot])
		results.append(TF.pil_to_tensor(pil_img).float() / NORM_FACTOR)
	new_data = normalize_augmentation((results, chosen_rots))
	return new_data


def is_augmentation(class_):
	return aug_name(class_) in LIST_OF_AUGS

def aug_name(class_):
	name_ = class_.split("_")[0]
	return name_

def augmentations_and_fns():
	dict_ = {}
	dict_['rand'] = rand_permutation
	dict_['noise'] = noise_augmentation
	dict_['horzflip'] = horzflip_augmentation
	dict_['crop'] = crop_augmentation
	dict_['rotation'] = rotation_augmentation
	return dict_

def apply_augs(main_class, data, batcher, class_list, aug_fns_dict):
	for aug in LIST_OF_AUGS:
		aug_name = '{}_{}'.format(aug, main_class)
		if aug_name in class_list:
			batcher[aug_name] = aug_fns_dict[aug](data)
