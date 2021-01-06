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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
from os import path
import pickle
import os
import pdb
from pprint import pprint
from copy import deepcopy


# NOTE [ldery] - put a freeze on adding anymore hyper-params until the ones you have are understood !
def add_trainer_args(parser):
	parser.add_argument('-train-epochs', type=int, default=50)
	parser.add_argument('-patience', type=int, default=50)
	parser.add_argument('-lr-patience', type=int, default=4)
	parser.add_argument('-optimizer', type=str, default='Adam')
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
					meta_split='train', alpha_update_algo='MoE'
				):
		self.chkpt_every = 10  # save to checkpoint
		self.train_epochs = train_epochs
		self.patience = patience
		self.max_grad_norm = 1.0
		self.meta_lr_weights = meta_lr_weights
		self.meta_lr_sgd = meta_lr_sgd
		self.meta_split = meta_split
		self.alpha_update_algo = alpha_update_algo
		print('Using {} as the update algorithm'.format(self.alpha_update_algo))

	def get_optim(self, model, opts, ft=False):
		lr = opts.lr if not ft else opts.ft_lr
		if opts.optimizer == 'Adam':
			optim = Adam(model.parameters(), lr=lr)
		elif opts.optimizer == 'SGD':
			optim = SGD(model.parameters(), lr=lr)
		elif opts.optimizer == 'AdamW':
			print('Using AdamW Optimizer')
			optim = AdamW(model.parameters(), lr=lr, weight_decay=0.1)
		else:
			raise ValueError
		# Reduce lr by 0.5 on plateau
		# save the learning rate
		self.lr = lr
		lr_scheduler = ReduceLROnPlateau(optim, factor=0.5, patience=opts.lr_patience, min_lr=1e-5)
		return optim, lr_scheduler

	def _get_numcorrect(self, output, targets, mask=None, chxnet_src=False):
		with torch.no_grad():
			argmax = output.argmax(dim=-1).squeeze()
			if output.shape[-1] == 1:  # We have only 1 output - so sigmoid.
				argmax = (torch.nn.Sigmoid()(output) > 0.5).squeeze().float()
			return argmax.eq(targets).sum()

	# Todo [ldery] - this function is no longer in use - either remove or re-write
	def run_model(self, x, y, model, group=None, reduct_='none'):
		model_out = model(x, head_name=group)
		loss_fn = model.get_loss_fn(model.loss_fn_name, reduction=reduct_)
		loss = loss_fn(model_out, y)
		num_correct = self._get_numcorrect(model_out, y)
		return loss, num_correct

	def run_epoch(self, model, data_iter, optim, alpha_generator=None, freeze_bn=False):
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
			total_loss = self.run_batch(model, batch, stats, alpha_generator)
			if optim is not None:
				total_loss.backward()  # backprop the accumulated gradient
				torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
				optim.step()

		summary = {}
		for k, v in stats.items():
			summary[k] = (v[0] / v[2], v[1].item() / v[2])
		return summary

	def run_batch(self, model, batch, stats, alpha_generator):
		loss_fn = model.get_loss_fn(model.loss_fn_name, reduction='mean')
		key_boundaries, prev_pos = {}, 0
		bulk_x = []
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
		for key, pos_pair in key_boundaries.items():
			this_out = shared_out[pos_pair[0]:pos_pair[1], :]
			head_name = key
			if ('rand' in key) or ('noise' in key):
				head_name = head_name.split('_')[-1]
			m_out = model(this_out, head_name=head_name, head_only=True)
			loss_ = loss_fn(m_out, pos_pair[-1])
			if alpha_generator is not None:
				alpha_sum += alpha_generator[key]
				total_loss = total_loss + alpha_generator[key] * loss_
			else:
				alpha_sum += 1.0
				total_loss += loss_
			if stats is not None:
				stats[key][0] += loss_.item() * len(pos_pair[-1])
				stats[key][1] += self._get_numcorrect(m_out, pos_pair[-1])
				stats[key][2] += len(pos_pair[-1])
		if isinstance(alpha_sum, torch.Tensor):
			alpha_sum = alpha_sum.item()
		return total_loss / alpha_sum

	def run_epoch_w_meta(self, model, group_iter, optim, primary_iter, meta_weights, primary_keys=None):
		# assert alpha_generator is not None, 'Need to specify a weight generating strategy'
		model.train()
		stats = defaultdict(lambda: [0.0, 0.0, 0.0])
		loss_fn = model.get_loss_fn(model.loss_fn_name, reduction='mean')
		primary_batches = [batch for batch in primary_iter]
		counter = 0
		if not hasattr(self, 'dp_stats'):
			self.dp_stats = defaultdict(list)
		start_ = len(self.dp_stats['people'])
		for batch in group_iter:
			# We do not have enough samples to continue, so wrap around
			counter = counter % len(primary_batches)
			if optim is not None:
				optim.zero_grad()

			# We need to learn the updated meta-weights here
			# Copy the model into a new model
			new_model = deepcopy(model)
			if self.alpha_update_algo == 'softmax':
				sm_meta_weights = get_softmax(meta_weights)
				loss_ = self.run_batch(new_model, batch, None, sm_meta_weights)
			else:
				loss_ = self.run_batch(new_model, batch, None, meta_weights)
			loss_.backward()

			# We can now take an SGD step here
			# Take a gradient step in the new model
			with torch.no_grad():
				for pname, param in new_model.named_parameters():
					if param.grad is None:
						continue
					# assert param.grad is not None
					#'All parameters of the new model should have filled gradients : {}'.format(pname)
					param.data.copy_(param.data - (self.meta_lr_sgd * param.grad.data))
					param.grad.zero_()

			# calculate the gradient w.r.t primary data
			prim_batch = primary_batches[counter]
			counter += 1
			for primary_key, (xs, ys) in prim_batch.items():
				assert 'rand' not in primary_key, 'The primary key is wrong : {}'.format(primary_key)
				results = self.run_model(xs, ys, new_model, group=primary_key, reduct_='mean')
				results[0].backward()

			# Now calculate the gradients of the meta-weights
			# Todo [ldery] - there is a memory-speed trade-off you can make here.
			if self.alpha_update_algo == 'softmax':
				sm_meta_weights = get_softmax(meta_weights)
			for b_idx, (key, (xs, ys)) in enumerate(batch.items()):
				if meta_weights[key].grad is None:
					meta_weights[key].grad = torch.zeros_like(meta_weights[key])
				meta_weights[key].grad.zero_()
				group = key
				if ('rand' in group) or ('noise' in group):
					group = group.split('_')[-1]
				result = self.run_model(xs, ys, model, group=group, reduct_='mean')
				grads = torch.autograd.grad(result[0], model.parameters(), allow_unused=True)
				dot_prod = 0.0
				with torch.no_grad():
					for idx_, (pname, param) in enumerate(new_model.named_parameters()):
						if grads[idx_] is None:
							assert 'fc' in pname, '{} - Module has none gradient which is invalid'.format(pname)
							continue
						if param.grad is None:
							continue
						dot_prod += (param.grad * grads[idx_]).sum()
					dot_prod = dot_prod.item()
				# Perform Gradient clipping here
				if self.alpha_update_algo == 'softmax':
					var_ = -sm_meta_weights[key] * dot_prod * self.meta_lr_sgd
					var_.backward(retain_graph=(b_idx != (len(batch) - 1)))
				else:
					var_ = -meta_weights[key] * dot_prod * self.meta_lr_sgd
					var_.backward()
				self.dp_stats[key].append(dot_prod)
			# update the meta_var
			with torch.no_grad():
				norm_val = 0.0
				for key in meta_weights.keys():
					if(self.alpha_update_algo == 'softmax') or (self.alpha_update_algo == 'linear') :
						new_val = meta_weights[key] - (self.meta_lr_weights * meta_weights[key].grad)
					else:
						assert '{} - Algo not implemented for updating alphas'.format(self.alpha_update_algo)
					norm_val += new_val.item()
					meta_weights[key].copy_(new_val)
					meta_weights[key].grad.zero_()
				if self.alpha_update_algo == 'linear':
					for k in meta_weights.keys():
						# Now do the projection
						normed_val = meta_weights[key] + (len(meta_weights) - norm_val) / len(meta_weights)
						meta_weights[key].copy_(normed_val)
				elif self.alpha_update_algo == 'MoE':
					for key in meta_weights.keys():
						meta_weights[key].div_(norm_val)
			total_loss = self.run_batch(model, batch, stats, meta_weights)
			if optim is not None:
				total_loss.backward()  # backprop the accumulated gradient
				torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
				optim.step()
			del new_model

		summary = {}
		for k, v in stats.items():
			summary[k] = (v[0] / v[2], v[1].item() / v[2])
# 		print({k: np.mean(v[-start_:]) for k, v in self.dp_stats.items()})
		return summary

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

	def train(self, model, dataset, optim, lr_scheduler=None, **kwargs):
		# Do the training
		# Check if the model already exists and just return it
		self.sanity_check_args(kwargs)
		self.metrics = defaultdict(lambda: defaultdict(list))
		best_val_acc = 0.0
		# saved_info = self.model_exists(model, dataset, kwargs)
		# if saved_info[0] is not None:
		# 	return saved_info[0], saved_info[1]

		chkpt_path = kwargs["model_chkpt_fldr"]
		best_path = os.path.join(chkpt_path, 'best.chkpt')
		last_path = os.path.join(chkpt_path, 'last.chkpt')
		monitor_list = kwargs['monitor_list']
		monitor_metric, best_epoch = [], -1
		# setup the meta-weights
		if kwargs['learn_meta_weights']:
			assert len(monitor_list) == 1, 'We can only learn meta-weights when there is 1 primary class'
# 			inits = np.random.normal(scale=2.0, size=len(kwargs['classes']))
			inits, idx_ = [], 0
			for idx, k in enumerate(kwargs['classes']):
				val_ = np.random.normal(size=1)[0]
				if k == 'people':
					idx_ = idx
				inits.append(val_)
			inits[idx_] -= 2.0
			inits = inits - min(inits) + (1.0 / len(kwargs['classes']))  # Make sure all are positive
			inits = inits / sum(inits)
			if self.alpha_update_algo == 'linear':
				num_classes = len(kwargs['classes'])
				inits = inits + (num_classes - sum(inits)) / num_classes
			meta_weights = {class_: torch.tensor([inits[id_]]).float().cuda() for id_, class_ in enumerate(kwargs['classes'])}
			for _, v in meta_weights.items():
				v.requires_grad = True
			if self.alpha_update_algo == 'softmax':
				this_weights = get_softmax(meta_weights)
				pprint(this_weights)
			else:
				pprint(meta_weights)
		to_eval = kwargs['classes']
		alpha_gen = kwargs['alpha_generator']
		m_weights = None
		
		if alpha_gen is not None and kwargs['learn_meta_weights']:
			this_weights = meta_weights
			if self.alpha_update_algo == 'softmax':
				this_weights = get_softmax(meta_weights)
			m_weights = {k: v.item() for k, v in this_weights.items()}
			alpha_gen.record_epoch_end(-1, 0, 0, meta_weights=m_weights)
		
		for i in range(self.train_epochs):
			tr_iter = dataset._get_iterator(kwargs['classes'], kwargs['batch_sz'], split='train', shuffle=True)
			if not kwargs['learn_meta_weights']:
				if alpha_gen is not None:
					alpha_gen.prep_epoch_start(i)
				tr_results = self.run_epoch(model, tr_iter, optim, alpha_generator=alpha_gen, freeze_bn=kwargs['freeze_bn'])
			else:
				primary_iter = dataset._get_iterator(monitor_list, kwargs['meta_batch_sz'], split=self.meta_split, shuffle=True)
				tr_results = self.run_epoch_w_meta(model, tr_iter, optim, primary_iter, meta_weights, primary_keys=monitor_list)
				this_weights = meta_weights
				if self.alpha_update_algo == 'softmax':
					this_weights = get_softmax(meta_weights)
				m_weights = {k: v.item() for k, v in this_weights.items()}

			for k, v in tr_results.items():
				self.metrics[k]['train'].append(v)
			val_iter = dataset._get_iterator(to_eval, kwargs['batch_sz'], split='val', shuffle=False)
			val_results = self.run_epoch(model, val_iter, None)
			to_avg = []
			for k, v in val_results.items():
				self.metrics[k]["val"].append(v)
				if k in monitor_list:
					to_avg.append(v[1])

			monitor_metric.append(np.mean(to_avg))
			if lr_scheduler is not None:
				lr_scheduler.step(monitor_metric[-1])

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
			print('--' * 30)

			if alpha_gen is not None:
				alpha_gen.record_epoch_end(i, monitor_metric[-1], np.mean(to_avg), meta_weights=m_weights)
			no_improvement = max(monitor_metric) not in monitor_metric[-self.patience:]
			if i > self.patience and no_improvement:
				break
		# Save the dot products for later analysis
		pickle.dump(self.dp_stats, open(os.path.join(chkpt_path, 'dp_stats.pkl'), 'wb'))
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
