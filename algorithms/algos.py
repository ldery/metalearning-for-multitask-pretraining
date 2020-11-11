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
import numpy as np
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
from os import path
import pickle
import os
import pdb
from pprint import pprint
from copy import deepcopy


def add_trainer_args(parser):
	parser.add_argument('-train-epochs', type=int, default=100)
	parser.add_argument('-patience', type=int, default=10)
	parser.add_argument('-lr-patience', type=int, default=4)
	parser.add_argument('-optimizer', type=str, default='Adam')
	parser.add_argument('-lr', type=float, default=3e-4)
	parser.add_argument('-meta-lr-weights', type=float, default=1e-1)
	parser.add_argument('-meta-lr-sgd', type=float, default=5e-2)  # Don't make this too high
	parser.add_argument('-batch-sz', type=int, default=128)
	parser.add_argument('-chkpt-path', type=str, default='experiments')
	parser.add_argument('-use-last-chkpt', action='store_true', help='Instead of using best, use last checkpoint')
	parser.add_argument('-continue-from-last', action='store_true')


class Trainer(object):
	def __init__(
					self, train_epochs, patience,
					meta_lr_weights=0.01, meta_lr_sgd=0.1
				):
		self.chkpt_every = 10  # save to checkpoint
		self.train_epochs = train_epochs
		self.patience = patience
		self.max_grad_norm = 1.0
		self.meta_lr_weights = meta_lr_weights
		self.meta_lr_sgd = meta_lr_sgd

	def get_optim(self, model, opts, ft=False):
		lr = opts.lr if not ft else opts.finetune_lr
		if opts.optimizer == 'Adam':
			optim = Adam(model.parameters(), lr=lr)
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

	def run_model(self, x, y, model, group=None, reduct_='none'):
		model_out = model(x, head_name=group)
		loss_fn = model.get_loss_fn(model.loss_fn_name, reduction=reduct_)
		loss = loss_fn(model_out, y)
		num_correct = self._get_numcorrect(model_out, y)
		return loss, num_correct

	def run_epoch(self, model, data_iter, optim):
		model.train()
		if optim is None:
			model.eval()
		stats = defaultdict(lambda: [0.0, 0.0, 0.0])
		for batch in data_iter:
			if optim is not None:
				optim.zero_grad()
			for key, (xs, ys) in batch.items():
				results = self.run_model(xs, ys, model, group=key, reduct_='mean')
				stats[key][0] += results[0].item() * len(ys)
				stats[key][1] += results[1].item()
				stats[key][2] += len(ys)
				results[0].backward()  # Accumulate the gradient so you don't hold on to graph

			if optim is not None:
				# torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
				optim.step()
		summary = {}
		for k, v in stats.items():
			summary[k] = (v[0] / v[2], v[1] / v[2])
		return summary

	def run_epoch_w_meta(self, model, group_iter, optim, primary_iter, meta_weights, primary_keys=None):
		assert primary_keys is not None
		model.train()
		stats = defaultdict(lambda: [0.0, 0.0, 0.0])
		group_batches = [batch for batch in group_iter]
		for iter_id, batch in enumerate(primary_iter):
			# This is quite a few short-cuts. Look to make this better
			if iter_id >= len(group_batches):
				break
			group_batch = group_batches[iter_id]

			optim.zero_grad()

			# clone the model
			new_model = deepcopy(model)

			# get the weighted losses
			n_losses = len(group_batch.keys())
			for key, (xs, ys) in group_batch.items():
				results = self.run_model(xs, ys, new_model, group=key, reduct_='mean')
				weighted_loss = (torch.sigmoid(meta_weights[key]).item()) * results[0]  # * (1.0 / n_losses)
				weighted_loss.backward()  # Accumulate the gradient so you don't hold on to graph

			# Take a gradient step in the new model
			with torch.no_grad():
				for pname, param in new_model.named_parameters():
					if param.grad is None:
						param.grad = torch.zeros_like(param)
						continue
					param.data.copy_(param.data - (self.meta_lr_sgd * param.grad.data))
					param.grad.zero_()

			# calculate the gradient w.r.t primary data
			for primary_key, (xs, ys) in batch.items():
				results = self.run_model(xs, ys, new_model, group=primary_key, reduct_='mean')
				stats[primary_key][0] += results[0].item() * len(ys)
				stats[primary_key][1] += results[1].item()
				stats[primary_key][2] += len(ys)
				results[0].backward()

			# Now calculate the gradients of the meta-weights
			# Todo [ldery] - there is a memory-speed trade-off you can make here.
			for key, (xs, ys) in group_batch.items():
				result = self.run_model(xs, ys, model, group=key, reduct_='mean')
				grads = torch.autograd.grad(result[0], model.parameters(), allow_unused=True)
				dot_prod = 0
				with torch.no_grad():
					for idx_, (pname, param) in enumerate(new_model.named_parameters()):
						if grads[idx_] is None:
							assert 'fc' in pname, '{} - Module has none gradient which is invalid'.format(pname)
							continue
						dot_prod += (param.grad * grads[idx_]).sum()

					dot_prod = dot_prod.item()
				var_ = self.meta_lr_sgd * (torch.sigmoid(meta_weights[key])) * dot_prod
				var_.backward()
				# update the meta_var
				with torch.no_grad():
					meta_weights[key].copy_(meta_weights[key] - self.meta_lr_weights * meta_weights[key].grad)
					meta_weights[key].grad.zero_()

			# update the gradients of the main model
			# get the weighted losses
			for key, (xs, ys) in group_batch.items():
				results = self.run_model(xs, ys, model, group=key, reduct_='mean')
				weighted_loss = torch.sigmoid(meta_weights[key]).item() * results[0]
				weighted_loss.backward()  # Accumulate the gradient so you don't hold on to graph

			# do an optimize step
			optim.step()

		summary = {}
		for k, v in stats.items():
			summary[k] = (v[0] / v[2], v[1] / v[2])
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
			inits = np.random.uniform(low=-1.0, high=1.0, size=len(kwargs['classes']))
			meta_weights = {class_: torch.tensor([inits[id_]]).float().cuda() for id_, class_ in enumerate(kwargs['classes'])}
			for _, v in meta_weights.items():
				v.requires_grad = True
		to_eval = kwargs['classes']
		for i in range(self.train_epochs):
			tr_iter = dataset._get_iterator(kwargs['classes'], kwargs['batch_sz'], split='train', shuffle=True)
			if not kwargs['learn_meta_weights']:
				tr_results = self.run_epoch(model, tr_iter, optim)
			else:
				primary_iter = dataset._get_iterator(monitor_list, kwargs['batch_sz'], split='train', shuffle=True)
				tr_results = self.run_epoch_w_meta(model, tr_iter, optim, primary_iter, meta_weights, primary_keys=monitor_list)
				m_weights = {k: torch.sigmoid(v).item() for k, v in meta_weights.items()}

			for k, v in tr_results.items():
				self.metrics[k]['train'].append(v)
			val_iter = dataset._get_iterator(to_eval, kwargs['batch_sz'], split='val', shuffle=False)
			val_results = self.run_epoch(model, val_iter, None)
			to_avg = []
			for k, v in val_results.items():
				self.metrics[k]["val"].append(v)
				if k in monitor_list:
					to_avg.append(v[1])

			test_iter = dataset._get_iterator(to_eval, kwargs['batch_sz'], split='test', shuffle=False)
			test_results = self.run_epoch(model, test_iter, None)

			for k, v in test_results.items():
				self.metrics[k]["test"].append(v)

			monitor_metric.append(np.mean(to_avg))
			if lr_scheduler is not None:
				lr_scheduler.step(monitor_metric[-1])

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

			no_improvement = max(monitor_metric) not in monitor_metric[-self.patience:]
			if i > self.patience and no_improvement:
				break
		# Load the best path
		model.load_state_dict(torch.load(best_path))
		test_iter = dataset._get_iterator(to_eval, kwargs['batch_sz'], split='test', shuffle=False)
		test_results = self.run_epoch(model, test_iter, None)
		print('Final Test Results - Best Epoch ', best_epoch)
		pprint({k: v for k, v in test_results.items() if k in monitor_list})
		return self.metrics, test_results
