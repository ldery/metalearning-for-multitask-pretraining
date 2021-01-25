from models.models import *
from data.datasets import *
from data.superclasses import *
from algorithms.algos import *
from algorithms.alpha_generator import add_weighter_args, get_alpha_generator
from argparse import ArgumentParser
import pdb
import random
import numpy as np
from pprint import pprint
from collections import defaultdict
import pickle
from copy import deepcopy


# Setup the arguments
def get_options():
	parser = ArgumentParser(description='Testbed for Datapoint Selection Code')\
	# Main Class
	parser.add_argument('-main-super-class', type=str, default='people')
	parser.add_argument('-log-comment', type=str, default='experiment')
	parser.add_argument('-seed', type=int, default=1)
	parser.add_argument('-num-runs', type=int, default=3, help='number of reruns so we can estimate confidence interval')
	parser.add_argument('-exp-name', default='test_run', type=str, help='name of this experiment')
	parser.add_argument('-data-seed', type=int, default=1234, help='The seed to use for the dataset')
	parser.add_argument('-mode', type=str, choices=['tgt_only', 'joint', 'pretrain', 'meta', 'pretrain_w_all'])
	parser.add_argument('-num-aux-tasks', type=int, default=4)
	# Self-supervised augmentations
	parser.add_argument('-use-random', action='store_true')
	parser.add_argument('-use-noise', action='store_true')
	parser.add_argument('-use-crop', action='store_true')
	parser.add_argument('-use-rotation', action='store_true')
	parser.add_argument('-use-horzflip', action='store_true')

	add_model_opts(parser)
	add_trainer_args(parser)
	add_weighter_args(parser)
	opts = parser.parse_args()
	return opts


def set_random_seed(seed):
	# Esp important for ensuring deterministic behavior with CNNs
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	cuda_available = torch.cuda.is_available()
	if cuda_available:
		torch.cuda.manual_seed_all(seed)
	return cuda_available


def train_model(
					algo, dataset, opts, seed, chosen_classes,
					monitor_classes, id_='direct_only', learn_meta_weights=False,
					model=None, primary_class=None, freeze_bn=False):
	set_random_seed(seed)
	chkpt_path = os.path.join('m4m_cache', opts.exp_name, id_, str(seed))
	if not os.path.exists(chkpt_path):
		os.makedirs(chkpt_path)
	all_classes = [*chosen_classes, *monitor_classes]
	out_class_dict = {}
	for chosen_key in all_classes:
		num_classes = NUM_PER_SUPERCLASS
		if 'rotation' in chosen_key:
			num_classes = 4
		out_class_dict[chosen_key] = num_classes

	ft = False
	use_last = False
	if model is None:
		batch_sz = opts.batch_sz
		model = WideResnet(
						out_class_dict, opts.depth, opts.widen_factor, dropRate=opts.dropRate,
					)
		model.cuda()
		use_last = opts.use_last_chkpt
	else:
		batch_sz = opts.ft_batch_sz
		ft = True
		all_classes = {k: NUM_PER_SUPERCLASS for k in all_classes}
		model.add_heads(all_classes)
	# Todo [ldery] - make sure future generators are compatible with multiple primary keys
	alpha_gen = None if primary_class is None else get_alpha_generator(opts, primary_class, chosen_classes)
	optim, lr_scheduler = algo.get_optim(model, opts, ft=ft)
	metrics, test_results = algo.train(
							model, dataset, optim, lr_scheduler=lr_scheduler,
							classes=chosen_classes, batch_sz=batch_sz,
							model_chkpt_fldr=chkpt_path, monitor_list=monitor_classes,
							learn_meta_weights=learn_meta_weights, alpha_generator=alpha_gen,
							meta_batch_sz=opts.meta_batch_sz, freeze_bn=freeze_bn, use_last_chkpt=use_last
						)
	if alpha_gen is not None:
		alpha_gen.viz_results(chkpt_path, group_aux=(not learn_meta_weights))
	return test_results, model
# Baselines :
# 	1. Model trained from scratch with no auxiliary data
# 	2. Multitasking with equalized auxiliary task weights
# 	3. Multitasking with meta-learned auxiliary task weights
# 	4. Multitasking with meta-learned data-parameters
#   4. Multitasking with data-augmentation

def add_ssl_tasks(main_superclass, chosen_set, opts):
	if opts.use_random:
		chosen_set.append('rand_{}'.format(main_superclass))
	if opts.use_noise:
		chosen_set.append('noise_{}'.format(main_superclass))
	if opts.use_crop:
		chosen_set.append('crop_{}'.format(main_superclass))
	if opts.use_rotation:
		chosen_set.append('rotation_{}'.format(main_superclass))
	if opts.use_horzflip:
		chosen_set.append('horzflip_{}'.format(main_superclass))

def main():
	opts = get_options()
	print(opts)
	opts.seed = opts.data_seed
	# Fix the dataset seed
	_ = set_random_seed(opts.seed)

	# Get the data
	dataset = CIFAR100(flatten=False)

	# Get the trainer
	result_dict = defaultdict(list)
	algo = Trainer(
				opts.train_epochs, opts.patience, meta_lr_weights=opts.meta_lr_weights,
				meta_lr_sgd=opts.meta_lr_sgd, meta_split=opts.meta_split,
				alpha_update_algo=opts.alpha_update_algo, use_cosine=(not opts.no_use_cosine),
				decoupled_weights=opts.decoupled_weights
			)
	chosen_set = list(cifar100_super_classes.keys())
	main_superclass = opts.main_super_class
	if chosen_set.index(main_super_class) < opts.num_aux_tasks:
		opts.num_aux_tasks += 1
	chosen_set = chosen_set[:opts.num_aux_tasks]

	if main_superclass not in chosen_set:
		chosen_set.append(main_superclass)
	add_ssl_tasks(main_superclass, chosen_set, opts)
	for seed in range(opts.num_runs):
		print('Currently on {}/{}'.format(seed + 1, opts.num_runs))
		set_random_seed(seed)
		if opts.mode == 'tgt_only':
			# 1. Model trained from scratch with no auxiliary data
			for main_class in chosen_set:
				this_id = "{}_only".format(main_class)
				chosen_classes, monitor_classes = [main_class], [main_class]
				this_res, _ = train_model(algo, dataset, opts, seed, chosen_classes, monitor_classes, id_=this_id)
				for k, v in this_res.items():
					result_dict[k].append(v[1])
		elif opts.mode == 'joint':
			# 2. Multitasking with equalized auxiliary task weights
			this_id = "joint_" + ".".join(chosen_set)
			monitor_classes, chosen_classes = chosen_set, chosen_set
			this_res, _ = train_model(algo, dataset, opts, seed, chosen_classes, monitor_classes, id_=this_id)
			for k, v in this_res.items():
				result_dict[k].append(v[1])
		elif opts.mode == 'pretrain':
			# 3. Pretrain with other tasks. Finetune on main task
			for main_class in chosen_set:
				if main_class != main_superclass:
					continue
				this_id = "pretr_{}".format(main_class)
				monitor_classes = [x for x in chosen_set if x != main_class]
				this_chosen = monitor_classes
				this_res, model = train_model(algo, dataset, opts, seed, this_chosen, monitor_classes, id_=this_id)

				# Now we do the training based on the class specific data.
				chosen_classes, monitor_classes = [main_class], [main_class]
				this_res, _ = train_model(algo, dataset, opts, seed, chosen_classes, monitor_classes, id_=this_id, model=model)
				for k, v in this_res.items():
					result_dict[k].append(v[1])
		elif opts.mode == 'pretrain_w_all':
			# 3. Pretrain all tasks. Finetune on main task
			this_id = "pretr_all"
			main_class = main_superclass
			monitor_classes = [main_class]
			this_res, model = train_model(
											algo, dataset, opts, seed, chosen_set,
											monitor_classes, id_=this_id, primary_class=main_class
										)
			for k, v in this_res.items():
				result_dict['pre_ft.{}'.format(k)].append(v[1])
			# for main_class in chosen_set:
				# Now we do the training based on the class specific data.
			new_model = deepcopy(model)
			chosen_classes, monitor_classes = [main_class], [main_class]
			this_res, _ = train_model(algo, dataset, opts, seed, chosen_classes, monitor_classes, id_=this_id, model=new_model)
			for k, v in this_res.items():
				result_dict[k].append(v[1])
		elif opts.mode == 'meta':
			# 4. Multitasking with meta-learned auxiliary task weights
			# for main_class in chosen_set:
			main_class = main_superclass
			monitor_classes = [main_class]
			this_id = "meta_{}".format(main_class)
			this_res, model = train_model(
								algo, dataset, opts, seed, chosen_set, monitor_classes,
								id_=this_id, learn_meta_weights=True, primary_class=main_class
							)
			for k, v in this_res.items():
				result_dict['pre_ft.{}'.format(k)].append(v[1])
			chosen_classes, monitor_classes = [main_class], [main_class]
			this_res, _ = train_model(
							algo, dataset, opts, seed, chosen_classes, monitor_classes,
							id_=this_id, model=model, freeze_bn=opts.freeze_bn
						)
			for k, v in this_res.items():
				result_dict[k].append(v[1])

			print('Results for epoch : {}/{}'.format(seed + 1, opts.num_runs))
			for k, v in result_dict.items():
				print("{:30s} = {:.3f} +/- {:.3f}".format(k, np.mean(v), np.std(v)))
			# Saving intermediate results since this takes quite some-time to run
			save_path = os.path.join('m4m_cache', opts.exp_name, "results.pkl")
			with open(save_path, 'wb') as handle:
				pickle.dump(result_dict, handle)

	for k, v in result_dict.items():
		print("{:30s} = {:.3f} +/- {:.3f}".format(k, np.mean(v), np.std(v)))
	save_path = os.path.join('m4m_cache', opts.exp_name, "results.pkl")
	with open(save_path, 'wb') as handle:
		pickle.dump(result_dict, handle)


if __name__ == '__main__':
	main()
