
# -*- coding:utf8 -*-
import numpy as np
import random
import torch
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.item_id = 0
        self.position = 0

        self.indice_y = [2, 3]
        self.indice_u = [4, 5]
        self.indice_c = [6, 7]
        self.indices_y_tensor = torch.LongTensor(self.indice_y)
        self.indices_u_tensor = torch.LongTensor(self.indice_u)
        self.indices_c_tensor = torch.LongTensor(self.indice_c)


    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, self.item_id)
        self.item_id += 1
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, step=None):
        if batch_size>=self.__len__():
            batch = self.buffer
        else:
            batch = sorted(random.sample(self.buffer, batch_size), key=lambda x : x[5])
        state, action, reward, next_state, done, item_id = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def normalizer_params(self):
        state = torch.FloatTensor(np.stack(np.array(self.buffer)[:, 0]))
        y_data = torch.index_select(state, -1, self.indices_y_tensor)
        u_data = torch.index_select(state, -1, self.indices_u_tensor)
        c_data = torch.index_select(state, -1, self.indices_c_tensor)
        r_data = torch.FloatTensor(np.stack(np.array(self.buffer)[:, 2]))

        def get_normalizer(x, x_name):
            if len(x.shape) == 3:
                x_mean = np.array(x.data).astype('float32').mean(axis=0)[0]
                x_std = np.array(x.data).astype('float32').std(axis=0)[0]
            elif len(x.shape) == 2 and x_name == 'r':
                x_mean = np.array(x.data).astype('float32').mean(axis=0)[0]
                x_std = np.array(x.data).astype('float32').std(axis=0)[0]
            else:
                x_mean = np.array(x.data).astype('float32').mean(axis=0)
                x_std = np.array(x.data).astype('float32').std(axis=0)

            return x_mean, x_std

        u_mean, u_std = get_normalizer(u_data, 'u')
        y_mean, y_std = get_normalizer(y_data, 'y')
        c_mean, c_std = get_normalizer(c_data, 'c')
        r_mean, r_std = get_normalizer(r_data, 'r')

        return y_mean, y_std, u_mean, u_std, c_mean, c_std, np.array([r_mean]), np.array([r_std])

    def save(self, save_folder):
        np.save(f"{save_folder}/replay_buffer.npy", self.buffer)


    def load(self, save_folder):
        self.buffer = np.load(f"{save_folder}/replay_buffer.npy", allow_pickle=True).tolist()
        self.item_id = self.buffer[-1][-1] + 1
        self.position = self.item_id

    def merge(self, replay_buffer):
        buffer_array = np.array(replay_buffer.buffer)
        buffer_array[:, -1] = buffer_array[:, -1] + self.item_id
        buffer = buffer_array.tolist()
        self.buffer = self.buffer + buffer

    def compute_return(self, gamma=0.99):
        pre_return = 0
        for i in range(len(self.buffer)):
            self.buffer[i] = list(self.buffer[i])
            self.buffer[i].insert(-3, self.buffer[i][2] + gamma * pre_return * (1 - self.buffer[i][4]))
            self.buffer[i] = tuple(self.buffer[i])
            pre_return = self.buffer[i][-1]


