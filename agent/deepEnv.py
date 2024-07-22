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
        self.beta_t = 1
        self.gTx = self.config.tx_gain
        self.gRx = self.config.rx_gain
        #look into the distribution of random arrays
        self.wavelength = self.generate_random_wavelength()
        self.d0 = self.generate_random_far_field_distance()
        self.path_loss_exponent = self.generate_random_path_loss_exponent()
        self.num_sbs = self.config.num_sbs
        self.num_ue = self.randomize_users()
        self.num_chnl = self.config.num_channel
        self.num_time_slots = 1
        self.min_sinr_threshold = self.config.min_sinr
        self.max_ue_per_sbs = self.config.max_num_usr
        self.sbs_state = np.ones(self.num_sbs)

        self.distance = self.randomize_distances()
        self.beta = 1
        
        self.g = self.calculate_channel_gain()
        self.alpha = self.allocate_channels_round_robin()
        
        self.p = self.randomize_transmission_power()
        self.noise_power = self.randomize_noise_power()

        self.B = self.config.band
        self.state = self.get_observation_state()
        self.action_space = Discrete(2)
       
        
    def step(self, action,count= 10):
        # Handle action (0: no handover, 1: handover)
        if action == 1:
            handover_successful = self.handover_users()
            if handover_successful:
                self.turn_off_idle_sbs()

        _, total_data_rate = self.calculate_data_rate()
        total_data_rate_mb = self.convert_datarate(total_data_rate)
        active_power, sleep_power, total_power = self.calculate_total_power()
        
        energy_efficiency = total_data_rate_mb / total_power
        reward = self.calculate_reward(energy_efficiency)
        done = np.all(self.calculate_done())
        
        if not done and count > 10:
            done = True

        self.state = self.get_observation_state()

        info = {
            'active_power': active_power,
            'sleep_power': sleep_power,
            'total_power': total_power,
            'energy_efficiency': energy_efficiency
        }
        return self.state, reward, done, info
        
        # Return step information
        # return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # get new state
        self.wavelength = self.generate_random_wavelength()
        self.d0 = self.generate_random_far_field_distance()
        self.path_loss_exponent = self.generate_random_path_loss_exponent()
        # self.num_ue = self.randomize_users()
        self.distance = self.randomize_distances()
        self.g = self.calculate_channel_gain()
        self.alpha = self.allocate_channels_round_robin()
        self.p = self.randomize_transmission_power()
        self.noise_power = self.randomize_noise_power()

        self.state =  self.get_observation_state() # All SBSs are initially on
        return self.state
    

    def generate_random_path_loss_exponent(self):
        # Typical range for path loss exponent in free space or ideal conditions
        min_n = 1.5
        max_n = 4.0
        random_n = random.uniform(min_n, max_n)
        return random_n

    def randomize_users(self):
        return [random.randint(self.config.min_num_usr, self.config.max_num_usr) for _ in range(self.num_sbs)]

    def randomize_distances(self):
        distances = []
        for num_ue in self.num_ue:
            distances.append(np.random.uniform(low=1.0, high=100.0, size=num_ue))
        return distances

    
    def generate_random_wavelength(self):
        # Speed of light in meters per second
        speed_of_light = 3e8  # 3 x 10^8 m/s
        # Define the frequency range in Hz (700 MHz to 5 GHz)
        min_frequency = 700e6  # 700 MHz in Hz
        max_frequency = 5e9    # 5 GHz in Hz
        # Generate a random frequency within this range
        random_frequency = random.uniform(min_frequency, max_frequency)
        # Calculate the wavelength using the formula λ = c / f
        wavelength = speed_of_light / random_frequency
        return wavelength
    
    def generate_random_far_field_distance(self):
        # Generate a random antenna dimension (D) between 0.1 meters and 3 meters
        min_dimension = 0.1  # meters
        max_dimension = 3.0  # meters
        random_dimension = random.uniform(min_dimension, max_dimension)
        # Generate a random wavelength
        wavelength = self.generate_random_wavelength()
        # Calculate the far-field distance using the formula R >= 2D^2 / λ
        far_field_distance = (2 * random_dimension**2) / wavelength
        return far_field_distance

    def randomize_transmission_power(self):
        transmission_powers = []
        for num_ue in self.num_ue:
            transmission_powers.append(np.random.uniform(low=0.1, high=1.0, size=num_ue))
        return transmission_powers
    
    def randomize_noise_power(self):
        noise_powers = []
        for num_ue in self.num_ue:
            noise_powers.append(np.random.uniform(low=1e-9, high=1e-7, size=num_ue))
        return noise_powers
        

    def calculate_channel_gain(self):
        gains = []
        for distances in self.distance:
            numerator = self.beta_t * self.gTx * self.gRx * (self.wavelength ** 2)
            denominator = 16 * (math.pi ** 2) * ((distances / self.d0) ** self.path_loss_exponent)
            gains.append(numerator / denominator)
        return gains
      
    def convert_datarate(self,datarate):
        """
        Convert bits to megabits.
        
        Parameters:
        bits (int or float): Number of bits to be converted.
        
        Returns:
        float: Number of megabits.
        """
        mb_arr = [bits / (1024 * 1024) for bits in datarate]
        return mb_arr
    
     
    def allocate_channels_round_robin(self):
        alpha = []
        for i, num_ue in enumerate(self.num_ue):
            ue_alpha = np.zeros((num_ue, self.num_chnl), dtype=int)
            for j in range(num_ue):
                r = j % self.num_chnl
                ue_alpha[j, r] = 1
            alpha.append(ue_alpha)
        return alpha

    def calculate_sinr(self,c,u,r):
        numerator = self.beta * self.p[c][u] * self.g[c][u]
        interference = 0.0

        for i in range(self.num_sbs):
            if i != c:
                for j in range(len(self.alpha[i])):
                    if j != u:
                        interference += self.beta * self.alpha[i][j, r] * self.p[i][j] * self.g[i][j]
        
        denominator = interference + self.noise_power[c][u]
        sinr = numerator / denominator

        #USE RAN Parser min and max for this one
        sinr = np.clip(sinr, 0, 20)
        return sinr


    def calculate_data_rate(self):
        data_rate = []
        # sbs_data_rate = []
        total_data_rate = np.zeros(self.num_sbs)

        for c in range(self.num_sbs):
            sbs_data_rate = np.zeros((len(self.alpha[c]), self.num_chnl))
            for u in range(len(self.alpha[c])):
                for r in range(self.num_chnl):
                    if self.alpha[c][u, r] == 1:
                        sinr = self.calculate_sinr(c, u, r)
                        sbs_data_rate[u,r] = self.B * np.log2(1 + sinr)
                        # sinr_val.append(sinr)
            data_rate.append(sbs_data_rate)
            total_data_rate[c] = np.sum(sbs_data_rate)
        return data_rate, total_data_rate

    def calculate_total_power(self):
        active_power = 0
        sleep_power = 0
        #need to recheck this logic for total power consumption for active mode and sleep mode
        for i in range(self.num_sbs):
            if np.sum(self.alpha[i]) > 0:
                active_power += self.sbs_state[i] * (np.sum(self.alpha[i] * self.p[i][:, np.newaxis]) + self.config.power_active)
            else:
                sleep_power += self.config.power_sleep
        total_power = active_power + sleep_power
        return active_power, sleep_power, total_power
    
    def generate_random_base_station_power(self):
        # Power ranges in watts
        min_power_watts = self.config.min_power  # Small cells and indoor base stations
        max_power_watts = self.config.max_power # High-power base stations
        
        # Generate a random power value in watts
        random_power_watts = np.random.uniform(min_power_watts, max_power_watts,size=(self.num_sbs))
        return random_power_watts
    

    #check this one as well if this or the above one makes more sense, which one makes more makes sense
    def calculate_base_station_power_consumption(self, efficiency=0.2, fixed_power=100):
        snr = self.calculate_sinr()
        # Calculate the signal power (S) using the SNR and noise power (N)
        signal_power = snr * self.randomize_noise_power()
        # Calculate the transmission power (P_tx) using the signal power (S) and channel gain (G)
        transmission_power = signal_power / self.calculate_channel_gain()
        # Calculate the total power consumption (P_total) of the base station
        total_power_consumption = (transmission_power / efficiency) + fixed_power
        return total_power_consumption

    def calculate_energy_efficiency(self):
        _, total_data_rate = self.calculate_data_rate()
        _,_,total_power = self.calculate_total_power()
        EE = total_data_rate / total_power
        return EE
    

    #check this reward function if it makes sense or is feasible 
    def calculate_reward(self, EE):
        Xj, Xk = self.check_constraints()
        U = sum(self.num_ue)
        C = self.num_sbs
        penalty = EE * ((1/U) * sum(Xj) + (1/C) * sum(Xk))
        return EE - penalty

    def check_constraints(self):
        Xj = np.zeros(sum(self.num_ue), dtype=int)
        Xk = np.zeros(self.num_sbs, dtype=int)
        print(self.num_ue)
        data_rate, _ = self.calculate_data_rate()
        for sbs in range(self.num_sbs):
            for ue in range(self.num_ue[sbs]):
                if data_rate[sbs][ue].sum() < self.config.demand_min:
                    Xj[ue] = 1
            if self.num_ue[sbs] > self.config.max_num_usr:
                Xk[sbs] = 1

        return Xj, Xk
    

    # def calculate_reward(self, total_data_rate_mb, total_power):
    #     power_factor = np.mean(total_power)  # mean power consumption
    #     performance_factor = np.mean(total_data_rate_mb)  # mean data rate
    #     reward = performance_factor / (power_factor + 1e-6)  # Add small value to avoid division by zero
    #     return reward
    
    def calculate_done(self):
        """
        Calculate the 'done' signal for each SBS based on the described logic.
        
        Returns:
        np.ndarray: Boolean array indicating whether 'done' is True or False for each SBS.
        """
        done = np.zeros(self.num_sbs, dtype=bool)
        for i in range(self.num_sbs):
            sinr = np.mean([self.calculate_sinr(i, u, r) for u in range(len(self.alpha[i])) for r in range(self.num_chnl) if self.alpha[i][u, r] == 1])
            done[i] = self.check_qos(sinr)
        return done
    
    def check_qos(self, sinr):
        """
        Check if all UEs associated with an SBS can meet their minimum required QoS based on SINR.
        Parameters:
        sinr (float): SINR value for the SBS.
        Returns:
        bool: True if all UEs can meet QoS, False otherwise.
        """
        # Replace with your QoS evaluation logic based on SINR
        return sinr >= self.min_sinr_threshold
    

    #need to look into the distance between UE and SBS for handing over as well. Only handover if it is in the minumum distance range.
    def handover_users(self):
        handover_successful = False
        for sbs_from in range(self.num_sbs):
            if self.sbs_state[sbs_from] == 1 and np.sum(self.alpha[sbs_from]) > 0:
                for u in range(len(self.alpha[sbs_from])):
                    if np.sum(self.alpha[sbs_from][u]) > 0:
                        for sbs_to in range(self.num_sbs):
                            if sbs_from != sbs_to and self.sbs_state[sbs_to] == 1:
                                if np.sum(self.alpha[sbs_to]) < self.max_ue_per_sbs:
                                    sinr_to = np.mean([self.calculate_sinr(sbs_to, u, r) for r in range(self.num_chnl) if u < len(self.alpha[sbs_to])])
                                    data_rate_to = np.mean([self.B * np.log2(1 + sinr_to)])
                                    if sinr_to >= self.min_sinr_threshold and data_rate_to >= self.config.demand_min:
                                        self.alpha[sbs_to][u] = self.alpha[sbs_from][u]
                                        self.alpha[sbs_from][u] = 0
                                        handover_successful = True
                                        break
                        if handover_successful:
                            break
        return handover_successful

    def turn_off_idle_sbs(self):
        for sbs in range(self.num_sbs):
            if np.sum(self.alpha[sbs]) == 0:
                self.sbs_state[sbs] = 0

    def get_observation_state(self):
        num_active_users = self.num_ue
        # get new state
        self.wavelength = self.generate_random_wavelength()
        self.d0 = self.generate_random_far_field_distance()
        self.path_loss_exponent = self.generate_random_path_loss_exponent()
        # self.num_ue = self.randomize_users()
        self.distance = self.randomize_distances()
        self.g = self.calculate_channel_gain()
        self.alpha = self.allocate_channels_round_robin()
        self.p = self.randomize_transmission_power()
        self.noise_power = self.randomize_noise_power()

        num_active_users_flat = np.array(num_active_users).flatten()
        data_rate, _ = self.calculate_data_rate()
        data_rate_mb = self.convert_datarate(data_rate)

        flattened_data = np.concatenate([array.flatten() for array in data_rate_mb])
        data_rate_flattened = flattened_data
        sinr_values = []
        
        sinr_values = []
        for c in range(self.num_sbs):
            sinr_sbs = np.zeros(self.num_ue[c], dtype=float)
            for u in range(self.num_ue[c]):
                sinr_sbs[u] = np.mean([self.calculate_sinr(c, u, r) for r in range(self.num_chnl)])
            sinr_values.append(sinr_sbs)
        sinr_values = np.concatenate(sinr_values)
        _, _, total_power = self.calculate_total_power()
        state =  np.concatenate((num_active_users_flat, data_rate_flattened, sinr_values, [total_power]))
        return state
