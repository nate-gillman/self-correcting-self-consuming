'''
File: /kin_policy.py
Created Date: Friday July 16th 2021
Author: Zhengyi Luo
Comment:
-----
Last Modified: Friday July 16th 2021 8:05:22 pm
Modified By: Zhengyi Luo at <zluo2@cs.cmu.edu>
-----
Copyright (c) 2021 Carnegie Mellon University, KLab
-----
'''

import torch.nn as nn
import torch
import pickle

from tqdm import tqdm

try:
    from uhc.khrylib.rl.core.distributions import DiagGaussian
    from uhc.khrylib.rl.core.policy import Policy
    from uhc.utils.math_utils import *
    from uhc.khrylib.models.mlp import MLP
    from uhc.khrylib.models.rnn import RNN
    from uhc.models import model_dict
    from uhc.utils.flags import flags
    from uhc.utils.torch_ext import get_scheduler
except:
    from UniversalHumanoidControl.uhc.khrylib.rl.core.distributions import DiagGaussian
    from UniversalHumanoidControl.uhc.khrylib.rl.core.policy import Policy
    from UniversalHumanoidControl.uhc.utils.math_utils import *
    from UniversalHumanoidControl.uhc.khrylib.models.mlp import MLP
    from UniversalHumanoidControl.uhc.khrylib.models.rnn import RNN
    from UniversalHumanoidControl.uhc.models import model_dict
    from UniversalHumanoidControl.uhc.utils.flags import flags
    from UniversalHumanoidControl.uhc.utils.torch_ext import get_scheduler 


import copy
from scipy.ndimage import gaussian_filter1d


class KinPolicy(Policy):
    def __init__(self, cfg, data_sample, device, dtype, mode="train"):
        super().__init__()
        self.cfg = cfg
        self.policy_specs = cfg.get("policy_specs", {})
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.type = 'gaussian'
        fix_std = cfg.policy_specs['fix_std']
        log_std = cfg.policy_specs['log_std']

        self.action_dim = action_dim = 80
        self.kin_net = model_dict[cfg.model_name](cfg,
                                                  data_sample=data_sample,
                                                  device=device,
                                                  dtype=dtype,
                                                  mode=mode)
        self.setup_optimizers()

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std,
                                           requires_grad=not fix_std)

        self.to(device)
        self.obs_lim = self.kin_net.get_obs(data_sample, 0)[0].shape[1]
        self.state_dim = state_dim = self.kin_net.state_dim
        self.mode = mode

        self.debug_qpos_ls = []

    def setup_optimizers(self):
        optim = self.cfg.get("policy_optimizer", "Adam")
        if optim == "Adam":
            print("Using Adam")
            self.optimizer = torch.optim.Adam(self.kin_net.parameters(),
                                              lr=self.cfg.lr)
        elif optim == "SGD":
            print("Using SGD")
            self.optimizer = torch.optim.SGD(self.kin_net.parameters(),
                                             lr=self.cfg.lr)
        elif optim == "Adamx":
            self.optimizer = torch.optim.Adamax(self.parameters(),
                                                lr=self.cfg.lr)
        else:
            raise NotImplementedError

        self.scheduler = get_scheduler(self.optimizer,
                                       policy="lambda",
                                       nepoch_fix=self.cfg.num_epoch_fix,
                                       nepoch=self.cfg.num_epoch)

    def init_context(self, data_dict, fix_height=True):
        """_summary_
        Initializing context for the policy
        Args:
            data_dict (_type_): _description_
            fix_height (bool, optional): _description_. Defaults to True.

        Raises:
            NotImplementedError: _description_
        """
        with torch.no_grad():
            pass

        raise NotImplementedError

    def step_lr(self):
        self.scheduler.step()

    def to(self, device):
        # ZL: annoying, need fix
        self.kin_net.to(device)
        super().to(device)
        self.device = device
        return self

    def reset(self):
        pass

    def reset_rnn(self, batch_size=1):
        self.kin_net.action_rnn.initialize(
            batch_size)  # initialize the RNN state again

    def set_mode(self, mode):
        self.mode = mode
        self.kin_net.set_mode(mode)
        if mode == "train":
            self.train()
        elif mode == "test":
            self.eval()


    def get_action(self, all_state):
        return self.kin_net.get_action(all_state)

    def forward(self, all_state):
        try:
            action_mean = self.get_action(all_state)
        except:
            print("Error in policy forward")
            import ipdb
            ipdb.set_trace()

        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return DiagGaussian(action_mean, action_std), action_mean, action_std

    def train_full_supervised(self,
                              cfg,
                              dataset,
                              device,
                              dtype,
                              num_epoch=20,
                              scheduled_sampling=0):
        pbar = tqdm(range(num_epoch))
        self.kin_net.set_schedule_sampling(scheduled_sampling)
        for epoch in pbar:
            train_loader = dataset.sampling_loader(num_samples=cfg.num_samples,
                                                   batch_size=cfg.batch_size,
                                                   num_workers=10,
                                                   fr_num=self.cfg.fr_num)
            self.kin_net.per_epoch_update(epoch)
            self.kin_net.training_epoch(train_loader, epoch)
        self.kin_net.eval_model(dataset)

    def update_init_supervised(self,
                               cfg,
                               dataset,
                               device,
                               dtype,
                               num_epoch=20):
        pbar = tqdm(range(num_epoch))
        for epoch in pbar:
            train_loader = dataset.sampling_loader(num_samples=cfg.num_samples,
                                                   batch_size=cfg.batch_size,
                                                   num_workers=10,
                                                   fr_num=self.cfg.fr_num)
            info = self.kin_net.train_first_frame_epoch(train_loader, epoch)
            pbar.set_description_str(
                f"Init loss: {info['total_loss'].cpu().detach().numpy():.3f}")

    def recrete_eps(self, data):
        masks, v_metas, _ = data
        device, dtype = masks.device, masks.dtype

        end_indice = np.where(masks.cpu().numpy() == 0)[0]
        v_metas = v_metas[end_indice, :]
        num_episode = len(end_indice)
        end_indice = np.insert(end_indice, 0, -1)
        max_episode_len = int(np.diff(end_indice).max())
        self.num_episode = num_episode
        self.max_episode_len = max_episode_len
        self.indices = np.arange(masks.shape[0])

        self.indices_end = np.arange(
            masks.shape[0] - num_episode)  # For choping off the last index
        for i in range(1, num_episode):
            start_index = end_indice[i] + 1
            end_index = end_indice[i + 1] + 1
            self.indices[
                start_index:end_index] += i * max_episode_len - start_index
            self.indices_end[(start_index -
                              i):(end_index - i -
                                  1)] += i * max_episode_len - start_index + i

        # self.s_scatter_indices = torch.LongTensor(np.tile(self.indices[:, None], (1, self.state_dim))).to(device)
        # self.s_gather_indices = torch.LongTensor(
        #     np.tile(self.indices[:, None], (1, self.state_dim))).to(device)

    def update_supervised(self,
                          all_state,
                          target_qpos,
                          curr_qpos,
                          num_epoch=20):
        pbar = tqdm(range(num_epoch))
        for _ in pbar:
            _, action_mean, _ = self.forward(all_state)
            self.kin_net.set_sim(curr_qpos)
            next_qpos, _ = self.kin_net.step(action_mean)
            loss, loss_idv = self.kin_net.compute_loss_lite(
                next_qpos, target_qpos)
            self.optimizer.zero_grad()
            loss.backward()  # Testing GT
            self.optimizer.step()  # Testing GT
            pbar.set_description_str(
                f"Per-step loss: {loss.cpu().detach().numpy():.3f} [{' '.join([str(f'{i * 1000:.3f}') for i in loss_idv])}] lr: {self.scheduler.get_lr()[0]:.5f}"
            )

            # total_loss, loss_dict, loss_unweighted_dict = self.kin_net.compute_loss_lite(next_qpos, target_qpos)
            # self.optimizer.zero_grad()
            # total_loss.backward()   # Testing GT
            # import ipdb; ipdb.set_trace()
            # self.optimizer.step()  # Testing GT
            # pbar.set_description_str(f"Per-step loss: {total_loss.cpu().detach().numpy():.3f} [{' '.join([str(f'{i * 1000:.3f}') for i in loss_unweighted_dict.values()])}] lr: {self.scheduler.get_lr()[0]:.5f}")

    def get_dim(self, x):
        dist = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(
            x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), dist.loc, {
            'std_id': std_id,
            'std_index': std_index
        }

    def select_action(self, x, mean_action=False):
        dist, action_mean, action_std = self.forward(x)

        action = action_mean if mean_action else dist.sample()
        return action

    def get_kl(self, x):
        dist, _, _ = self.forward(x)
        return dist.kl()

    def get_log_prob(self, x, action):
        dist, _, _ = self.forward(x)
        return dist.log_prob(action)
