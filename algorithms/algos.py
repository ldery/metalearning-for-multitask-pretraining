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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from collections import defaultdict
from os import path
import pickle
import os
import pdb
from pprint import pprint
from copy import deepcopy
import higher


# NOTE [ldery] - put a freeze on adding anymore hyper-params until the ones you have are understood !
def add_trainer_args(parser):
	parser.add_argument('-train-epochs', type=int, default=100)
	parser.add_argument('-patience', type=int, default=30)
	parser.add_argument('-lr-patience', type=int, default=4)
	parser.add_argument('-optimizer', type=str, default='Adam')
	parser.add_argument('-ft-optimizer', type=str, default='Adam')
	parser.add_argument('-lr', type=float, default=1e-3)
	parser.add_argument('-ft-lr', type=float, default=5e-5)
	parser.add_argument(
		'-meta-lr-weights', type=float, default=1e-3,
		help='Outer-loop lr for weights. Very important to tune'
	) # If using cosine instead of dot prod - cross validate in log-space
	# Don't make this too high if not the consecutive grads don't align much
	parser.add_argument('-meta-lr-sgd', type=float, default=1e-2, help='Inner loop sgd lr. Very important to tune')
	parser.add_argument('-batch-sz', type=int, default=320)
	parser.add_argument('-ft-batch-sz', type=int, default=128)
	parser.add_argument(
							'-meta-batch-sz', type=int, default=64,
							help='Size of the meta-batch to use. This is actually really important'
							', using larger values reduces the update variance and makes algo more stable'
	)
	parser.add_argument('-chkpt-path', type=str, default='experiments')
	parser.add_argument('-use-last-chkpt', action='store_true', help='Instead of using best, use last checkpoint')
	parser.add_argument('-continue-from-last', action='store_true')
	parser.add_argument('-meta-split', type=str, default='val', choices=['train', 'val'])
	parser.add_argument('-freeze-bn', action='store_true', help='Freeze the BN layers after pre-training')
	parser.add_argument('-alpha-update-algo', type=str, default='MoE', choices=['softmax', 'MoE', 'linear'])
	parser.add_argument('-no-use-cosine', action='store_true', help='Do not use cosine instead of dot product')
	parser.add_argument('-decoupled-weights', action='store_true', help='Decouple the norm from the proportion of the auxiliary task')
	parser.add_argument('-use-lr-scheduler', action='store_true', help='Whether to use an lr scheduler')


def get_softmax(weights):
	keys = []
	values = []
	for k, v in weights.items():
		keys.append(k)
		values.append(v)

	joint_vec = torch.cat(values)
	softmax = F.softmax(joint_vec, dim=-1)
	return {k: v for k, v in zip(keys, softmax)}


class Trainer(object):
	def __init__(
					self, train_epochs, patience,
					meta_lr_weights=0.01, meta_lr_sgd=0.1,
					meta_split='train', alpha_update_algo='MoE',
					use_cosine=True, decoupled_weights=False, use_scheduler=False
				):
		self.chkpt_every = 10  # save to checkpoint
		self.train_epochs = train_epochs
		self.patience = patience
		self.max_grad_norm = 1.0
		self.meta_lr_weights = meta_lr_weights
		self.meta_lr_sgd = meta_lr_sgd
		self.meta_split = meta_split
		self.alpha_update_algo = alpha_update_algo
		self.use_cosine = use_cosine
		self.decoupled_weights = decoupled_weights
		self.inner_iters = 1 # Todo [ldery] change this into a hyper-parameter that is passed in
		self.use_scheduler = use_scheduler
		print('Using {} as the update algorithm'.format(self.alpha_update_algo))

	def get_optim(self, model, opts, ft=False):
		if not ft:
			lr = opts.lr
			optimizer = opts.optimizer
		else:
			lr = opts.ft_lr
			optimizer = opts.ft_optimizer
		print('Using Optimizer : ', optimizer)
		if optimizer == 'Adam':
			optim = Adam(model.parameters(), lr=lr)
		elif optimizer == 'SGD':
			# Not including momentum so that the behavior is predictable.
			# Todo [ldery] - change this when final experimentation is ready
			optim = SGD(model.parameters(), lr=lr, momentum=0.0)
		elif optimizer == 'AdamW':
			print('Using AdamW Optimizer')
			optim = AdamW(model.parameters(), lr=lr, weight_decay=0.1)
		else:
			raise ValueError
		self.lr = lr
		lr_scheduler = StepLR(optim, self.train_epochs // 3, gamma=0.1, last_epoch=-1)
		return optim, lr_scheduler

	def _get_numcorrect(self, output, targets):
		with torch.no_grad():
			argmax = output.argmax(dim=-1).squeeze()
			if output.shape[-1] == 1:  # We have only 1 output - so sigmoid.
				argmax = (torch.nn.Sigmoid()(output) > 0.5).squeeze().float()
			return argmax.eq(targets).sum()

	def run_model(self, x, y, model, group=None, reduct_='none'):
		model_out = model(x, head_name=group)
		loss_fn = model.get_loss_fn(model.loss_fn_name, reduction=reduct_)
		loss = loss_fn(model_out, y)
		num_correct = self._get_numcorrect(model_out, y)
		return loss, num_correct

	def run_epoch(self, model, data_iter, optim, alpha_generator=None, class_norms=None, freeze_bn=False):
		# assert alpha_generator is not None, 'Need to specify a weight generating strategy'
		model.train()
		if optim is None:
			model.eval()
		elif freeze_bn:
			# We are fine-tuning so freeze the B.N layers
			for module in model.modules():
				if isinstance(module, nn.BatchNorm2d):
					if hasattr(module, 'weight'):
						module.weight.requires_grad_(False)
					if hasattr(module, 'bias'):
						module.bias.requires_grad_(False)
					module.eval()

		stats = defaultdict(lambda: [0.0, 0.0, 0.0])
		for batch in data_iter:
			if optim is not None:
				optim.zero_grad()
			total_loss = self.run_batch(model, batch, stats, alpha_generator, class_norms)
			if optim is not None:
				total_loss.backward()  # backprop the accumulated gradient
				torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
				optim.step()

		summary = {}
		for k, v in stats.items():
			summary[k] = (v[0] / v[2], v[1].item() / v[2])
		return summary

	# An optimized version that allows us to batch together data from different tasks
	# This has implications for the gradient - so use with caution
	def run_batch(self, model, batch, stats, alpha_generator, class_norms):
		loss_fn = model.get_loss_fn(model.loss_fn_name, reduction='mean')
		key_boundaries, prev_pos = {}, 0
		bulk_x = []
		# [ldery] - the key boundary function has been tested via pdb traces
		# Consolidate all data for different tasks into 1 batch
		for key, (xs, ys) in batch.items():
			# gather the batches together. Skip if
			if (alpha_generator is not None) and alpha_generator[key] == 0.0:
				continue
			bulk_x.append(xs)
			key_boundaries[key] = (prev_pos, prev_pos + len(xs), ys)
			prev_pos += len(xs)
		# Stack and run through model
		joint_x = torch.cat(bulk_x, axis=0)
		shared_out = model(joint_x, body_only=True)
		total_loss = 0
		alpha_sum = 0.0
		# Now run individual task batches through their respective model-heads.
		for key, pos_pair in key_boundaries.items():
			this_out = shared_out[pos_pair[0]:pos_pair[1], :]
			head_name = key
			loss_mul_fact = 1.0
			if ('rand' in key) or ('noise' in key) or ('negloss' in key):
				head_name = head_name.split('_')[-1]
				loss_mul_fact = -loss_mul_fact if 'negloss' in key else loss_mul_fact
			m_out = model(this_out, head_name=head_name, head_only=True)
			loss_ = loss_mul_fact * loss_fn(m_out, pos_pair[-1])

			if stats is not None:
				stats[key][0] += loss_.item() * len(pos_pair[-1])
				stats[key][1] += self._get_numcorrect(m_out, pos_pair[-1])
				stats[key][2] += len(pos_pair[-1])

			if alpha_generator is not None:
				total_loss = total_loss + alpha_generator[key] * loss_ * class_norms[key]
				alpha_sum += alpha_generator[key]
			else:
				alpha_sum += 1.0
				total_loss += loss_
		if isinstance(alpha_sum, torch.Tensor):
			alpha_sum = alpha_sum.item()
		return total_loss / alpha_sum

	# todo [ldery] - maybe move these to utils.py
	def calc_norm(self, grads):
		norm = 0.0
		for g_ in grads:
			if g_ is not None:
				norm += (g_**2).sum()
		return np.sqrt(norm.item())
	
	def dot_prod(self, g1, g2):
		total = 0.0
		for p1, p2 in zip(g1, g2):
			if p1 is None or p2 is None:
				continue
			total += (p1 * p2).sum()
		return total.item()

	# Does training that meta-learns the task weights
	def run_epoch_w_meta(self, model, group_iter, optim, primary_iter, meta_weights, class_norms, primary_keys=None):
		model.train()
		stats = defaultdict(lambda: [0.0, 0.0, 0.0])
		loss_fn = model.get_loss_fn(model.loss_fn_name, reduction='mean')
		# Convert batch iterators to list. Todo [ldery] - make this more efficient in the future
		primary_batches = [batch for batch in primary_iter]
		group_iter = [batch for batch in group_iter]
		# Allow re-using the meta-val set if we run out.
		prim_idxs = np.random.choice(len(primary_batches), size=len(group_iter), replace=True)

		# Collect statistics for post-hoc analysis
		if not hasattr(self, 'dp_stats'):
			self.dp_stats = defaultdict(list)
			start_ = 0
		else:
			start_ = len(self.dp_stats[list(self.dp_stats.keys())[0]])
		if not hasattr(self, 'weight_stats'):
			self.weight_stats = defaultdict(list)

		for group_idx, batch in enumerate(group_iter):

			optim.zero_grad()

			# Take the inner-loop step
			# This is set to 0.0 so we have no look-ahead step
			override_dict = {'lr': [torch.tensor([0.0]).cuda()]}
			task_grads = {}
			with higher.innerloop_ctx(model, optim, track_higher_grads=True, override=override_dict) as (fmodel, diffopt):
				# Learn the updated model
				this_weights = get_softmax(meta_weights) if self.alpha_update_algo == 'softmax' else meta_weights

				total_loss, alpha_sum = 0.0, 0.0
				for k, v in batch.items():
					loss_ = loss_fn(fmodel(v[0], head_name=k), v[1])
					task_grads[k] = torch.autograd.grad(loss_, fmodel.parameters(), retain_graph=True, allow_unused=True)

					# Now apply the weighting
					loss_ = loss_ * this_weights[k] * class_norms[k]
					alpha_sum += (this_weights[k] * class_norms[k])
					total_loss += loss_

				prim_batch = primary_batches[prim_idxs[group_idx]]
				# Compute the loss on the meta-val set
				for primary_key, (xs, ys) in prim_batch.items():
					assert 'rand' not in primary_key, 'The primary key is wrong : {}'.format(primary_key)
					results = self.run_model(xs, ys, fmodel, group=primary_key, reduct_='mean')

				# For computing statistics
				meta_grad = torch.autograd.grad(results[0], fmodel.parameters(), retain_graph=True, allow_unused=True)
				meta_norm = self.calc_norm(meta_grad)


			del fmodel
			del diffopt

			# Compute appropriate normalization factors
			# Also collect statistics
			for key, _ in batch.items():
				grads = task_grads[key]
				key_norm = self.calc_norm(grads)
				dot_prod = self.dot_prod(meta_grad, grads)

				# Apply appropriate normalization and save statistics
				with torch.no_grad():
					if self.use_cosine:
						normalization = key_norm * meta_norm
					else:
						normalization = self.meta_lr_sgd
					if self.decoupled_weights and class_norms[key].grad is not None:
						class_norms[key].grad.div_(normalization)
					assert meta_weights[key].grad is None, 'This should be none at the moment'
					meta_weights[key].grad = torch.zeros_like(meta_weights[key]) - dot_prod
					meta_weights[key].grad.div_(normalization)
					self.dp_stats[key].append(dot_prod)
					self.weight_stats[key].append((meta_weights[key].item(), meta_weights[key].grad.item(), meta_norm, key_norm, self.dp_stats[key][-1]))


			# Update the task level weightings
			with torch.no_grad():
				self.update_meta_weights(meta_weights, update_algo=self.alpha_update_algo)
				if self.decoupled_weights:
					self.update_meta_weights(class_norms, update_algo='class_linear')
				this_weights = get_softmax(meta_weights) if self.alpha_update_algo == 'softmax' else meta_weights

			# Now take a true gradient step with the corrected weights
			total_loss = self.run_batch(model, batch, stats, this_weights, class_norms)
			if optim is not None:
				total_loss.backward()  # backprop the accumulated gradient
				torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
				optim.step()

			# Reset the gradients of the scalings
			for k, v in meta_weights.items():
				v.grad = None
				class_norms[k].grad = None

		summary = {}
		for k, v in stats.items():
			summary[k] = (v[0] / v[2], v[1].item() / v[2])

		try:
			res = (np.array(list(self.dp_stats.values())) > 0)*1.0
		except:
			pdb.set_trace()
		print({k: np.mean(v[-start_:]) for k, v in self.dp_stats.items()}, res.sum(axis=-1))
		return summary
	
	# Perform update on meta-related weights.
	def update_meta_weights(self, meta_weights, update_algo='softmax'):
		ABS_MIN = 1e-6
		norm_val = 0.0
		for key in meta_weights.keys():
			if meta_weights[key].grad is None:
				meta_weights[key].grad = torch.zeros_like(meta_weights[key])
			if(update_algo == 'softmax') or ('linear' in update_algo) :
				new_val = meta_weights[key] - (self.meta_lr_weights * meta_weights[key].grad)
			else:
				assert '{} - Algo not implemented for updating alphas'.format(self.alpha_update_algo)
			if update_algo == 'linear':
				new_val = torch.clamp(new_val, ABS_MIN, len(meta_weights))
			elif update_algo == 'class_linear':
				new_val = new_val * (new_val > 0.0)
			norm_val += new_val.item()
			meta_weights[key].copy_(new_val)
			meta_weights[key].grad.zero_()
		if update_algo == 'linear':
			# won't be called when update algo is class_linear
			for key in meta_weights.keys():
				# Now do the projection
				normed_val = meta_weights[key] + (len(meta_weights) - norm_val) / len(meta_weights)
				meta_weights[key].copy_(normed_val * (normed_val > 0.0))
		elif update_algo == 'MoE':
			for key in meta_weights.keys():
				meta_weights[key].div_(norm_val)

	# ldery - consider removing since not used
	def model_exists(self, model, dataset, kwargs):
		chkpt_path = kwargs["model_chkpt_fldr"]
		dir_name = path.dirname(chkpt_path)
		metric_path = path.join(dir_name, "train_metrics.pkl")
		if not path.exists(metric_path):
			return None, None
		train_metrics = pickle.load(open(metric_path, 'rb'))
		best_val_loss = min(train_metrics['monitor_loss'][-(self.patience + 1):])
		final_path = self.get_chkpt_path('best', chkpt_path)
		if path.exists(final_path):
			model.load_state_dict(torch.load(final_path))
			return train_metrics, best_val_loss
		else:
			return None, None

	def get_chkpt_path(self, epoch, chkpt_path):
		this_path = chkpt_path.split('.')[:-1]
		this_path = ".".join(this_path)
		return this_path + "_epoch_" + str(epoch) + ".chkpt"

	def sanity_check_args(self, arg_dict):
		assert 'classes' in arg_dict, 'List of classes not in arguments'
		assert 'batch_sz' in arg_dict, 'Arg dict must have batch size specified'
		assert 'model_chkpt_fldr' in arg_dict, 'Need to specify where to checkpoint model'
		assert 'monitor_list' in arg_dict, 'Need to specify which super-classes are used for val monitoring'

	def print_metrics(self, epoch, metric_dict):
		final_str = ""
		for k, v in metric_dict.items():
			this_str = "{:30s} | ".format(k)
			str_ = ""
			for type_, vals in v.items():
				str_ = "{} = ({:.3f}, {:.3f})| {}".format(type_, *(vals[-1]), str_)
			this_str += str_
			final_str = "{}\n{}".format(this_str, final_str)
		print('Epoch {}\n'.format(epoch))
		print(final_str)
	
	def create_weights(self, classes, init='random', norm=1.0, requires_grad=True):
		if init == 'ones':
			inits = np.array([1.0]* len(classes))
		elif init == 'random':
			inits = np.random.normal(scale=2.0, size=len(classes))
			inits = inits - min(inits) + (1.0 / len(classes))  # Make sure all are positive
		elif init == 'mins':
			inits = np.array([1e-7]*len(classes))

		if norm > 0:
			inits = norm * inits / (sum(inits))
		# Create the weights
		weights = {class_: torch.tensor([inits[id_]]).float().cuda() for id_, class_ in enumerate(classes)}
		if requires_grad:
			for _, v in weights.items():
				v.requires_grad = True
		return weights

	def train(self, model, dataset, optim, lr_scheduler=None, **kwargs):
		# Do the training
		# Check if the model already exists and just return it
		self.sanity_check_args(kwargs)
		self.metrics = defaultdict(lambda: defaultdict(list))
		best_val_acc = 0.0

		chkpt_path = kwargs["model_chkpt_fldr"]
		best_path = os.path.join(chkpt_path, 'best.chkpt')
		last_path = os.path.join(chkpt_path, 'last.chkpt')
		monitor_list = kwargs['monitor_list']
		monitor_metric, best_epoch = [], -1
		# setup the class norms : 
		class_norms = self.create_weights(
											kwargs['classes'], init='ones', norm=len(kwargs['classes']),
											requires_grad=self.decoupled_weights
										)

		# setup the meta-weights
		if kwargs['learn_meta_weights']:
			assert len(monitor_list) == 1, 'We can only learn meta-weights when there is at least 1 primary class'
			norm = 1.0 if self.alpha_update_algo == 'softmax' else len(kwargs['classes'])
			meta_weights = self.create_weights(kwargs['classes'], init='ones', norm=norm)
			if self.alpha_update_algo == 'softmax':
				this_weights = get_softmax(meta_weights)
				pprint(this_weights)
			else:
				pprint(meta_weights)
		to_eval = kwargs['classes']
		alpha_gen = kwargs['alpha_generator']
		m_weights = None

		# Mark the first epoch
		if alpha_gen is not None and kwargs['learn_meta_weights']:
			this_weights = meta_weights
			if self.alpha_update_algo == 'softmax':
				this_weights = get_softmax(meta_weights)
			m_weights = {k: v.item() for k, v in this_weights.items()}
			c_weights = {k: v.item() for k, v in class_norms.items()}
			alpha_gen.record_epoch_end(-1, 0, 0, meta_weights=m_weights, class_norms=c_weights)

		# Train the model
		for i in range(self.train_epochs):
			tr_iter = dataset._get_iterator(kwargs['classes'], kwargs['batch_sz'], split='train', shuffle=True)
			if not kwargs['learn_meta_weights']:
				if alpha_gen is not None:
					alpha_gen.prep_epoch_start(i)
				tr_results = self.run_epoch(model, tr_iter, optim, alpha_generator=alpha_gen, class_norms=class_norms, freeze_bn=kwargs['freeze_bn'])
			else:
				primary_iter = dataset._get_iterator(monitor_list, kwargs['meta_batch_sz'], split=self.meta_split, shuffle=True)
				tr_results = self.run_epoch_w_meta(model, tr_iter, optim, primary_iter, meta_weights, class_norms, primary_keys=monitor_list)
				this_weights = meta_weights
				if self.alpha_update_algo == 'softmax':
					this_weights = get_softmax(meta_weights)
				m_weights = {k: v.item() for k, v in this_weights.items()}

			for k, v in tr_results.items():
				self.metrics[k]['train'].append(v)

			# Gather the statistics for Test and Validation
			val_iter = dataset._get_iterator(to_eval, kwargs['batch_sz'], split='val', shuffle=False)
			val_results = self.run_epoch(model, val_iter, None)
			to_avg = []
			for k, v in val_results.items():
				self.metrics[k]["val"].append(v)
				if k in monitor_list:
					to_avg.append(v[1])

			monitor_metric.append(np.mean(to_avg))
			if self.use_scheduler:
				lr_scheduler.step()
				self.meta_lr_sgd = optim.state_dict()['param_groups'][0]['lr']
				print('Using Stepped LR : ', self.meta_lr_sgd)

			test_iter = dataset._get_iterator(to_eval, kwargs['batch_sz'], split='test', shuffle=False)
			test_results = self.run_epoch(model, test_iter, None)
			to_avg = []
			for k, v in test_results.items():
				self.metrics[k]["test"].append(v)
				if k in monitor_list:
					to_avg.append(v[1])

			if monitor_metric[-1] >= best_val_acc:
				# We have seen an improvement in validation loss
				best_val_acc = monitor_metric[-1]
				torch.save(model.state_dict(), best_path)
				best_epoch = i
			else:
				torch.save(model.state_dict(), last_path)

			self.print_metrics(i, self.metrics)
			if kwargs['learn_meta_weights']:
				pprint(m_weights)
			print('These are the class norms')
			pprint(class_norms)
			print('--' * 30)

			if alpha_gen is not None:
				c_weights = {k: v.item() for k, v in class_norms.items()}
				alpha_gen.record_epoch_end(i, monitor_metric[-1], np.mean(to_avg), meta_weights=m_weights, class_norms=c_weights)
			no_improvement = max(monitor_metric) not in monitor_metric[-self.patience:]
			if i > self.patience and no_improvement:
				break
		# Save the dot products and metrics for later analysis
		if kwargs['learn_meta_weights']:
			pickle.dump(self.dp_stats, open(os.path.join(chkpt_path, 'dp_stats.pkl'), 'wb'))
			pickle.dump({k:v for k, v in self.metrics.items()}, open(os.path.join(chkpt_path, 'metrics.pkl'), 'wb'))
			pickle.dump({k:v for k, v in self.weight_stats.items()}, open(os.path.join(chkpt_path, 'weight_stats.pkl'), 'wb'))

		# Load the best path
		if kwargs['use_last_chkpt']:
			model.load_state_dict(torch.load(last_path))
		else:
			model.load_state_dict(torch.load(best_path))
		test_iter = dataset._get_iterator(to_eval, kwargs['batch_sz'], split='test', shuffle=False)
		test_results = self.run_epoch(model, test_iter, None)
		print('Final Test Results - Best Epoch ', best_epoch)
		pprint({k: v for k, v in test_results.items() if k in monitor_list})
		return self.metrics, test_results
