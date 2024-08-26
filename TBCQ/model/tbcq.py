#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import os

import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from TBCQ.control.base_offline import *
from TBCQ.common.normalize import NoNormalizer

from .common import *
from .common import DiagMultivariateNormal as MultivariateNormal
from .func import normal_differential_sample, multivariate_normal_kl_loss, zeros_like_with_shape


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.phi = phi

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], -1)))
        a = F.relu(self.l2(a))
        a = self.phi * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], -1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], -1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], -1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

# Variational Recurrent Neural Network + Vanilla VAE
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, tl, device, net_type='GRU', num_layers=1):
        super(VAE, self).__init__()

        self.k = 128
        self.latent_dim = latent_dim
        self.observation_size = state_dim
        self.latent_size = latent_dim
        self.input_size = action_dim
        self.num_layers = num_layers
        if net_type == 'rnn':
            RnnClass = torch.nn.RNN
        elif net_type == 'GRU':
            RnnClass = torch.nn.GRU
        else:
            raise NotImplementedError

        self.rnn = RnnClass(3 * self.k, self.k, num_layers=num_layers)
        self.process_u = PreProcess(self.input_size, self.k)
        self.process_x = PreProcess(self.observation_size, self.k)
        self.process_z = PreProcess(self.latent_size, self.k)
        self.process_e = PreProcess(self.latent_size, self.k)

        self.posterior_gauss = DBlock(2 * self.k, 3 * self.k, self.latent_size)
        self.prior_gauss = DBlock(self.k, 3 * self.k, self.latent_size)
        self.decoder = DBlock(2 * self.k, 3 * self.k, self.observation_size)

        #  vanilla vae
        self.e1 = nn.Linear(4 * self.k, 750)
        self.e2 = nn.Linear(750, 750)
        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)
        self.d1 = nn.Linear(4 * self.k, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.tl = tl

        self.device = device

    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):

        external_input_seq_embed = self.process_u.forward(external_input_seq)
        observations_seq_embed = self.process_x.forward(observations_seq)

        l, batch_size, _ = external_input_seq.size()

        h, rnn_hidden_state = (
            zeros_like_with_shape(external_input_seq, (batch_size, self.k)),
            zeros_like_with_shape(external_input_seq, (self.num_layers, batch_size, self.k))
        ) if memory_state is None else (memory_state['hn'], memory_state['rnn_hidden'])

        z_t = zeros_like_with_shape(external_input_seq, (batch_size, self.k))

        state_mu = []
        state_logsigma = []
        sampled_state = []
        h_seq = [h]
        rnn_hidden_state_seq = [rnn_hidden_state.transpose(1, 0)]
        for t in range(l):
            # Estimate the posterior distribution of z at each moment t
            z_t_mean, z_t_logsigma = self.posterior_gauss.forward(
                torch.cat([observations_seq_embed[t], h], dim=-1)
            )
            # Sampling from distribution to obtain z_t
            z_t = normal_differential_sample(
                MultivariateNormal(z_t_mean, logsigma2cov(z_t_logsigma))
            )

            # Update h and the hidden state of RNN
            output, rnn_hidden_state = self.rnn(torch.cat(
                [observations_seq_embed[t], external_input_seq_embed[t], self.process_z.forward(z_t)], dim=-1
            ).unsqueeze(dim=0), rnn_hidden_state)
            h = output[0]

            # record
            state_mu.append(z_t_mean)
            state_logsigma.append(z_t_logsigma)
            sampled_state.append(z_t)
            h_seq.append(h)
            rnn_hidden_state_seq.append(rnn_hidden_state.contiguous().transpose(1, 0))

        # list stack
        state_mu = torch.stack(state_mu, dim=0)
        state_logsigma = torch.stack(state_logsigma, dim=0)
        sampled_state = torch.stack(sampled_state, dim=0)
        h_seq = torch.stack(h_seq, dim=0)
        rnn_hidden_state_seq = torch.stack(rnn_hidden_state_seq, dim=0)

        # vanilla vae
        z_t_embed = self.process_z.forward(z_t)
        e_t = F.relu(self.e1(torch.cat([h_seq[-2], observations_seq_embed[-1], external_input_seq_embed[-1], z_t_embed], dim=-1)))
        e_t = F.relu(self.e2(e_t))

        e_t_mean = self.mean(e_t)
        # Clamped for numerical stability
        e_t_log_std = self.log_std(e_t).clamp(-4, 15)
        e_t_std = torch.exp(e_t_log_std)
        e_t = e_t_mean + e_t_std * torch.randn_like(e_t_std)
        a_t = self.decode_external_input(h_seq[-2], e_t, observations_seq_embed[-1], z_t_embed)

        outputs = {
            'a_t': a_t,
            'e_t': e_t,
            'e_t_mean': e_t_mean,
            'e_t_std': e_t_std,
            'state_mu': state_mu,
            'state_logsigma': state_logsigma,
            'sampled_state': sampled_state,
            'h_seq': h_seq,
            'external_input_seq_embed': external_input_seq_embed,
            'rnn_hidden_state_seq': rnn_hidden_state_seq
        }
        return outputs, {'hn': h, 'rnn_hidden': rnn_hidden_state}

    def call_loss(self, external_input_seq, observations_seq, memory_state=None):
        outputs, memory_state = self.forward_posterior(external_input_seq, observations_seq, memory_state)

        h_seq = outputs['h_seq']
        state_mu = outputs['state_mu']
        state_logsigma = outputs['state_logsigma']
        a_t_recon = outputs['a_t']
        e_t_mean = outputs['e_t_mean']
        e_t_std = outputs['e_t_std']

        l, batch_size, _ = observations_seq.shape

        predicted_h = h_seq[:-1]

        # vrnn loss
        prior_z_t_seq_mean, prior_z_t_seq_logsigma = self.prior_gauss.forward(
            predicted_h
        )

        z_kl_loss = multivariate_normal_kl_loss(
            state_mu,
            logsigma2cov(state_logsigma),
            prior_z_t_seq_mean,
            logsigma2cov(prior_z_t_seq_logsigma)
        )

        observations_normal_sample = self.decode_observation(outputs, mode='sample')
        x_recon_loss = F.mse_loss(observations_normal_sample, observations_seq)

        vrnn_loss = z_kl_loss + x_recon_loss

        # vanilla vae loss
        a_recon_loss = F.mse_loss(a_t_recon, external_input_seq[-1])
        e_kl_loss = -0.5 * (1 + torch.log(e_t_std.pow(2)) - e_t_mean.pow(2) - e_t_std.pow(2)).mean()
        vanilla_vae_loss = a_recon_loss + e_kl_loss

        return vrnn_loss, vanilla_vae_loss, a_recon_loss

    def decode_observation(self, outputs, mode='sample'):
        mean, logsigma = self.decoder.forward(
            torch.cat([
                self.process_z.forward(outputs['sampled_state']), outputs['h_seq'][:-1]
            ], dim=-1)
        )
        observations_normal_dist = MultivariateNormal(
            mean, logsigma2cov(logsigma)
        )
        if mode == 'dist':
            return observations_normal_dist
        elif mode == 'sample':
            return observations_normal_dist.sample()

    def decode_external_input(self, h=None, e=None, x_embed=None, z_embed=None):
        if e is None:
            e = torch.randn((x_embed.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        e_embed = self.process_e.forward(e)
        a = F.relu(self.d1(torch.cat([h, e_embed, x_embed, z_embed], dim=-1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def predict(self, state, action):
        external_input_seq = action
        observations_seq = state[:-1]

        outputs, memory_state = self.forward_posterior(external_input_seq, observations_seq)

        h_seq = outputs['h_seq']
        sampled_state = outputs['sampled_state']
        e_t = outputs['e_t']

        z_t_embed = self.process_z.forward(sampled_state[-1])
        x_seq_embed = self.process_x.forward(state)

        predict_action = self.decode_external_input(h_seq[-2], e_t, x_seq_embed[-1], z_t_embed)

        return predict_action


class TBCQ(OfflineBase):
    def __init__(self,
                 y_size,
                 u_size,
                 c_size,
                 u_bounds=None,

                 replay_buffer=None,
                 normalizer=None,
                 gpu_id=0,

                 max_u=2,
                 discount=0.99,
                 tau=0.005,
                 lmbda=0.75,
                 phi=0.05,
                 lr=1e-3,

                 batch_size=64,

                 ts=True,
                 tl=10,

                 logging=None
                 ):

        assert ts is True, "TSCQ only support time series data"
        super(TBCQ, self).__init__(y_size, u_size, c_size, gpu_id=gpu_id, u_bounds=u_bounds, replay_buffer=replay_buffer)
        self.model = None        # model free RL

        self.normalizer = normalizer if normalizer else NoNormalizer()
        self.replay_buffer = replay_buffer

        self.batch_size = batch_size
        self.tl = tl

        # Definition of Network
        state_dim, action_dim, c_dim, max_action = 2 * y_size + c_size, u_size, c_size, max_u
        latent_dim = action_dim * 2

        # actor network
        self.actor = Actor(state_dim, action_dim, max_action, phi).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # critic network
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # VAE network
        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, tl, self.device).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # coefficient
        self.discount = discount        # target Q
        self.tau = tau                  # Update Target Networks
        self.lmbda = lmbda              # soft-clip

        self.logging = logging

        self.indices_y_tensor = torch.LongTensor(self.indice_y).to(self.device)
        self.indices_c_tensor = torch.LongTensor(self.indice_c).to(self.device)
        self.indices_y_star_tensor = torch.LongTensor(self.indice_y_star).to(self.device)

        self.history_state = None
        self.history_action = None

    def _act(self, state):
        y = self.normalizer.normalize(state[self.indice_y], 'y')
        y_star = self.normalizer.normalize(state[self.indice_y_star], 'y')
        c = self.normalizer.normalize(state[self.indice_c], 'c')
        state = torch.FloatTensor(np.hstack((y, y_star, c))).to(self.device).unsqueeze(0)

        # history info
        if self.history_state is None:
            self.history_state = [list(copy.deepcopy(state).detach().cpu().numpy())[0] for _ in range(self.tl)]
        if self.history_action is None:
            self.history_action = [list(np.zeros(self.action_dim)) for _ in range(self.tl)]

        with torch.no_grad():
            state_list = torch.Tensor(self.history_state).unsqueeze(1).to(self.device)
            action_list = torch.Tensor(self.history_action).unsqueeze(1).to(self.device)

            predicted_action = self.vae.predict(state_list, action_list[1:])

            action = self.actor(
                state.repeat_interleave(100, dim=0),
                predicted_action.repeat_interleave(100, dim=0))  # repeat batch_size times
            q1 = self.critic.q1(state.repeat_interleave(100, dim=0), action)
            ind = q1.argmax(0)

        act = action[ind].detach().cpu().numpy().flatten()
        act = self.normalizer.inverse(act, 'u')
        act = np.clip(act, self.u_bounds[:, 0], self.u_bounds[:, 1])  # bounds act in u_bounds

        # update history info
        action_record = self.normalizer.normalize(act, 'u')
        self.history_state.pop(0)
        self.history_state.append(list(state.detach().cpu().numpy())[0])
        self.history_action.pop(0)
        self.history_action.append(action_record)

        return act

    def _train(self, s, u, ns, r, done):

        self.logging(f"step: {self.step} \ts: {s} \tu: {u} \tns: {ns} \t penalty: {r} \tdone: {done} ")
        self.train()

    def train(self, iterations):

        start_time = time.time()

        critic_loss_seq = []
        actor_loss_seq = []
        vrnn_loss_seq = []
        vanilla_vae_loss_seq = []
        vanilla_vae_recon_loss_seq = []
        vae_loss_seq = []
        vae_recon_loss_seq = []

        for it in range(int(iterations)):

            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
            done = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)

            # Recombination of state variables
            state = torch.FloatTensor(self.normalizer.normalize(state, 'yyuc')).to(self.device)
            next_state = torch.FloatTensor(self.normalizer.normalize(next_state, 'yyuc')).to(self.device)
            action = torch.FloatTensor(self.normalizer.normalize(action, 'u')).to(self.device)
            reward = -torch.FloatTensor(self.normalizer.normalize(reward, 'r')).unsqueeze(-1).to(self.device)

            # Extract characteristic data items from the state
            y = torch.index_select(state, -1, self.indices_y_tensor)
            ny = torch.index_select(next_state, -1, self.indices_y_tensor)
            y_star = torch.index_select(state, -1, self.indices_y_star_tensor)
            ny_star = torch.index_select(next_state, -1, self.indices_y_star_tensor)
            c = torch.index_select(state, -1, self.indices_c_tensor)
            nc = torch.index_select(next_state, -1, self.indices_c_tensor)

            # New state reorganization
            state = torch.cat((y, y_star, c), dim=-1)
            next_state = torch.cat((ny, ny_star, nc), dim=-1)

            # Dimension transformation - to adapt to the time series length tl
            state = state.permute(1, 0, 2)
            next_state = next_state.permute(1, 0, 2)
            action = action.permute(1, 0, 2)
            reward = reward.permute(1, 0, 2)
            done = done.permute(1, 0, 2)

            # VAE Training
            vrnn_loss, vanilla_vae_loss, vanilla_vae_recon_loss = self.vae.call_loss(action, state)
            vrnn_loss_seq.append(vrnn_loss.detach().cpu())
            vanilla_vae_loss_seq.append(vanilla_vae_loss.detach().cpu())
            vanilla_vae_recon_loss_seq.append(vanilla_vae_recon_loss.detach().cpu())

            vae_loss = vanilla_vae_loss + vrnn_loss
            vae_loss_seq.append(vae_loss.detach().cpu())

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            # Critic Training
            with torch.no_grad():
                # Duplicate next state 10 times
                next_state = torch.repeat_interleave(next_state, 10, dim=1)
                history_action = torch.repeat_interleave(action, 10, dim=1)

                target_Q1, target_Q2 = self.critic_target(
                    next_state[-1], self.actor_target(next_state[-1], self.vae.predict(next_state, history_action[1:]))
                )

                # Soft Clipped Double Q-learning
                target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)

                # Take max over each action sampled from the VAE
                target_Q = target_Q.reshape(self.batch_size, -1).max(-1)[0].reshape(-1, 1)
                target_Q = reward[-1] + (1 - done[-1]) * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(state[-1], action[-1])
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            critic_loss_seq.append(critic_loss.detach().cpu())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Pertubation Model / Action Training
            predicted_action = self.vae.predict(state, action[:-1])
            perturbed_actions = self.actor(state[-1], predicted_action)

            vae_recon_loss = F.mse_loss(predicted_action, action[-1]).detach().cpu()
            vae_recon_loss_seq.append(vae_recon_loss)

            # Update through DPG
            actor_loss = -self.critic.q1(state[-1], perturbed_actions).mean()
            actor_loss_seq.append(actor_loss.detach().cpu())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        end_time = time.time()

        self.logging('vae_loss:%5f' % float(np.mean(vae_loss_seq)),
                     'vae_recon_loss:%5f' % float(np.mean(vae_recon_loss_seq)),
                     'vrnn_loss:%5f' % float(np.mean(vrnn_loss_seq)),
                     'vanilla_vae_loss:%5f' % float(np.mean(vanilla_vae_loss_seq)),
                     'vanilla_vae_recon_loss:%5f' % float(np.mean(vanilla_vae_recon_loss_seq)),
                     ' critic_loss:%5f' % float(np.mean(critic_loss_seq)),
                     ' actor_loss:%5f' % float(np.mean(actor_loss_seq)),
                     ' sum_loss:%5f' % float(np.mean(vrnn_loss_seq) + np.mean(vae_loss_seq) + np.mean(critic_loss_seq) + np.mean(actor_loss_seq)),
                     ' time_used:%5f' % (end_time - start_time),
                     )

        return vae_loss_seq, critic_loss_seq, actor_loss_seq
