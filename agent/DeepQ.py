from gym import Env
from gym.spaces import Discrete, Box
import random
import math
import numpy as np
from RANParser import RANParser

class CarrierEnv(Env):
    def __init__(self):
        self.parser = RANParser()
        self.config = self.parser.parse_args()
        
        self.num_sbs = self.config.num_sbs
        self.num_ue = self.config.num_usr
        self.num_chnl = self.config.num_channel
        self.num_time_slots = 1

        self.beta_t = self.config.beta_t
        self.gTx = self.config.tx_gain
        self.gRx = self.config.rx_gain
        self.wavelength = self.config.wavelength
        self.d0 = self.config.d0
        self.path_loss_exponent = self.config.path_loss
        
        self.B = self.config.band
        self.p_sleep = self.config.power_sleep
        self.p_active = self.config.power_active

        self.distance = self.randomize_distances()
        self.beta = self.config.beta_t
        self.g = self.calculate_channel_gain()
        self.alpha = self.allocate_channels_round_robin()
        self.p = self.randomize_transmission_power()
        self.noise_power = self.randomize_noise_power()

        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=1, shape=(self.num_sbs,), dtype=np.float32)

        self.state = np.ones(self.num_sbs, dtype=np.float32)  # All SBSs are initially on

    def step(self, action):
        # Randomize variables for the current time step
        self.distance = self.randomize_distances()
        self.p = self.randomize_transmission_power()
        self.noise_power = self.randomize_noise_power()
        self.g = self.calculate_channel_gain()
        
        data_rate, total_data_rate = self.calculate_data_rate()
        total_power = self.calculate_total_power()
        energy_efficiency = total_data_rate / total_power

        reward = energy_efficiency - np.sum(self.state) / self.num_sbs  # Penalize active SBSs

        done = True
        info = {}
        
        return self.state, reward, done, info

    def render(self):
        pass
    
    def reset(self):
        self.state = np.ones(self.num_sbs, dtype=np.float32)  # All SBSs are initially on
        return self.state

    def randomize_distances(self):
        return np.random.uniform(low=1.0, high=100.0, size=(self.num_sbs, self.num_ue))

    def randomize_transmission_power(self):
        return np.random.uniform(low=0.1, high=1.0, size=(self.num_sbs, self.num_ue, self.num_chnl))

    def randomize_noise_power(self):
        return np.random.uniform(low=1e-9, high=1e-7)  # Random noise power within a given range

    def calculate_channel_gain(self):
        numerator = self.beta_t * self.gTx * self.gRx * (self.wavelength ** 2)
        denominator = 16 * (math.pi ** 2) * ((self.distance / self.d0) ** self.path_loss_exponent)
        signal_strength = numerator / denominator
        return signal_strength

    def allocate_channels_round_robin(self):
        alpha = np.zeros((self.num_sbs, self.num_ue, self.num_chnl), dtype=int)
        for i in range(self.num_sbs):
            for j in range(self.num_ue):
                r = (j + i) % self.num_chnl
                alpha[i, j, r] = 1
        return alpha

    def calculate_sinr(self):
        c, u, r = 0, 0, 0
        numerator = self.beta * self.p[c, u, r] * self.g[c, u, r]
        interference = 0.0
        C, U, R = self.alpha.shape

        for i in range(C):
            if i != c:
                for j in range(U):
                    if j != u:
                        interference += self.beta * self.alpha[i, j, r] * self.p[i, j, r] * self.g[i, j, r]

        denominator = interference + self.noise_power
        sinr = numerator / denominator
        return sinr

    def calculate_data_rate(self):
        SINR = self.calculate_sinr()
        C, U, R = self.alpha.shape
        data_rate = np.zeros((C, U, R))

        for c in range(C):
            for u in range(U):
                for r in range(R):
                    if self.alpha[c, u, r] == 1:
                        data_rate[c, u, r] = self.B * np.log2(1 + SINR)
        
        total_data_rate = np.sum(data_rate, axis=(1, 2))
        return data_rate, total_data_rate

    def calculate_total_power(self):
        total_power = np.zeros(self.num_time_slots)
        for t in range(self.num_time_slots):
            for i in range(self.num_sbs):
                active_power = self.beta * (np.sum(self.alpha[i, :, :] * self.p[i, :, :]) + self.p_active)
                sleep_power = (1 - self.beta) * self.p_sleep
                total_power[t] += active_power + sleep_power
        return total_power

    def calculate_energy_efficiency(self):
        _, total_data_rate = self.calculate_data_rate()
        total_power = self.calculate_total_power()
        EE = total_data_rate / total_power
        return EE

    def calculate_reward(self, Xj, Xk):
        EE = self.calculate_energy_efficiency()
        U = self.num_ue
        C = self.num_sbs
        return EE - EE * ((1/U) * np.sum(Xj) + (1/C) * np.sum(Xk))
