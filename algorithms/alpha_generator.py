from abc import abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import pdb


def add_weighter_args(parser):
	parser.add_argument(
							'-weight-strgy', type=str, default='default',
							choices=['default', 'alt', 'warm_up_down']
	)
	# add arguments for alternating
	parser.add_argument('-alt-freq', type=int, default=1, help='If using alt strategy, how often to alternate')
	parser.add_argument('-prim-start', type=int, default=0, help='What epoch to start training on the primary loss')
	parser.add_argument('-init-val', type=float, default=1.0, help='Initial Task weightings')
	parser.add_argument('-end-val', type=float, default=1.0, help='Final task weightings')


def get_alpha_generator(opts, prim_key, aux_keys):
	weight_strgy = opts.weight_strgy
	if weight_strgy == 'default':
		return DefaultWeighter(prim_key, aux_keys, init_val=opts.init_val)
	elif weight_strgy == 'alt':
		return AlternatingWeighter(
					prim_key, aux_keys, init_val=opts.init_val,
					alt_freq=opts.alt_freq, prim_start=opts.prim_start
				)
	elif weight_strgy == 'warm_up_down':
		return WarmUpAndDown(
					prim_key, aux_keys, init_val=opts.init_val,
					prim_first=(opts.prim_start == 0), end_val=opts.end_val,
					reset_every=opts.alt_freq
				)
	else:
		assert 'Invalid Value for Weighting strategy - {}'.format(weight_strgy)


class Weighter(object):
	def __init__(self, prim_key, aux_keys, init_val=1.0):
		self.weights = {key: init_val for key in aux_keys}
		self.aux_keys = aux_keys
		self.weights[prim_key] = init_val
		self.prim_key = prim_key
		self.init_val = init_val
		self.result_logs = []

	def __getitem__(self, key):
		return self.weights[key]

	@abstractmethod
	def prep_epoch_start(self, epoch, **kwargs):
		pass

	def record_epoch_end(self, epoch, stat, **kwargs):
		if kwargs['meta_weights'] is None:
			entry = [v for _, v in self.weights.items()]
		else:
			entry = [v for _, v in kwargs['meta_weights'].items()]
		# Place the statistic to record in the final position
		entry.append(stat)
		self.result_logs.append(entry)

	def viz_results(self, save_loc, metric_name='Test Accuracy', group_aux=True):
		to_viz = np.array(self.result_logs)
		all_keys = list(self.weights.keys())
		prim_idx = all_keys.index(self.prim_key)
		prim_vals = to_viz[:, prim_idx]
		fig, ax = plt.subplots(figsize=(16, 8))
		ax.plot(range(len(prim_vals)), prim_vals, label='Primary Task Weighting')
		for idx_, key in enumerate(all_keys):
			if idx_ == prim_idx:
				continue
			desc = '{}'.format(key) if not group_aux else 'Weight Per-Auxiliary Task'
			ax.plot(range(len(prim_vals)), to_viz[:, idx_], linestyle='dashed', label=desc)
			if group_aux:
				break
		ax.set_xlabel('Epoch')
		ax.set_ylabel('Weighting')
		ax.legend()
		ax2 = ax.twinx()
		ax2.plot(range(len(prim_vals)), to_viz[:, -1], color='tab:red', label=metric_name)
		ax2.set_ylabel(metric_name, color='tab:red')
		plt.savefig('{}/weighting_vrs_stat.png'.format(save_loc))


class DefaultWeighter(Weighter):
	def __init__(self, prim_key, aux_keys, init_val=1.0):
		super(DefaultWeighter, self).__init__(prim_key, aux_keys, init_val=init_val)

	def __getitem__(self, key):
		return self.init_val

	def prep_epoch_start(self, epoch, **kwargs):
		pass


class AlternatingWeighter(Weighter):
	def __init__(self, prim_key, aux_keys, init_val=1.0, alt_freq=1, prim_start=0):
		super(AlternatingWeighter, self).__init__(prim_key, aux_keys, init_val=init_val)
		self.alt_freq = alt_freq
		self.prim_start = prim_start

	def prep_epoch_start(self, epoch, **kwargs):
		# restore the initial weighting
		if (epoch < self.prim_start) or ((epoch - self.prim_start) % self.alt_freq) == 0:
			prim_val = 0.0
			aux_val = self.init_val
		else:
			aux_val = 0.0
			prim_val = self.init_val
		self.weights = {k: aux_val for k in self.aux_keys}
		self.weights[self.prim_key] = prim_val


class WarmUpAndDown(Weighter):
	def __init__(self, prim_key, aux_keys, init_val=0.0, prim_first=True, end_val=1.0, reset_every=5):
		super(WarmUpAndDown, self).__init__(prim_key, aux_keys, init_val=init_val)
		self.end_val = end_val
		self.reset_every = reset_every
		self.prim_first = prim_first
		self.delta = (end_val - init_val) / self.reset_every
		assert self.delta > 0, 'End - {} is less than start - {}'.format(end_val, init_val)

	def prep_epoch_start(self, epoch, **kwargs):
		# We are starting the epoch.
		eff_epoch = epoch // self.reset_every
		aux_delt, prim_delt = self.delta, self.delta
		if self.prim_first:
			if eff_epoch % 2 == 0:
				# We are in a phase where primary weight is dropping
				prim_delt = -self.delta
			else:
				aux_delt = -self.delta
		else:
			# We are in a phase where auxiliary weight is dropping
			if eff_epoch % 2 == 0:
				aux_delt = -self.delta
			else:
				prim_delt = -self.delta
		assert prim_delt * aux_delt < 0, 'prim_delt = {} and aux_delt = {} are same sign'.format(prim_delt, aux_delt)
		if epoch % self.reset_every == 0:
			# We are resetting
			self.weights[self.prim_key] = self.end_val if prim_delt < 0 else self.init_val
			for k, v in self.weights.items():
				if k == self.prim_key:
					continue
				self.weights[k] = self.end_val if aux_delt < 0 else self.init_val
		else:
			# We are doing a normal update
			old_val = self.weights[self.prim_key]
			self.weights = {k: self.weights[k] + aux_delt for k in self.aux_keys}
			self.weights[self.prim_key] = old_val + prim_delt

