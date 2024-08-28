import argparse
import datetime
import os
from os.path import join as pjoin
import math


class RANParser(argparse.ArgumentParser):
    TIME_STAMP = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')

    @staticmethod
    def __home_out(path):
        full_path = pjoin('./' , 'ran', 'data', path)
       # print("Aaaaa")
        if not os.path.exists(full_path):
            #print("Makedir")
            os.makedirs(full_path)
        return full_path

    def __init__(self):
        super(RANParser, self).__init__()
        self.__init_ssb()
        self.__init_dqn()
        self.__init_agent()
        self.__init_env()

    def __init_ssb(self):
        self.add_argument('--num_rrh', type=int, default=8, help='number of RRH per cell')

        self.add_argument('--demand_min', type=float, default=250, help='minimal user demand mbps per cell')
        self.add_argument('--demand_max', type=float, default=600, help='maximal user demand mbps per cell')
        self.add_argument('--max_num_usr', type=int, default=20, help='max number of usr associated per cell')
        self.add_argument('--min_num_usr', type=int, default=5, help='minimum number of usr associated per cell')


        self.add_argument('--total_user', type=int,default=5,help="The fixed number of user associated in the whole network")
        self.add_argument('--min_distance', type=int,default=5,help="Minimum distance for handover intilisation")
        self.add_argument('--max_distance', type=int,default=30,help="Maximum distance for handover intilisation")
        # self.add_argument('--distance',type=float,default=23.6,help='distance between UE i and SBS c at time t')

        # self.add_argument('--beta_t',type=float,default=1.0,help='A scaling factor specific to the scenario, possibly including system losses or gains.')
        self.add_argument('--tx_gain',type=float,default=12.0,help='Transmit antenna gain.')
        self.add_argument('--rx_gain',type=float,default=10.0,help='Received antenna gain.')
        self.add_argument('--wavelength',type=float,default=0.05,help='Wavelength of the signal.')

        # self.add_argument('--interference',type=float,default=0.3,help='Interference level: 0.3 is 30%.')
        self.add_argument('--path_loss',type=float,default=2,help='Path-loss exponent.')
        self.add_argument('--diameter',type=float,default=1.2,help='Diameter of the antenna.')


        self.add_argument('--num_sbs',type=int,default=2,help='the number of SBS')
        self.add_argument('--num_channel',type=int,default=3,help='the number of channel')

        self.add_argument('--num_usr', type=int, default=5, help='number of usr associated per cell')

        self.add_argument('--noise_power',type=float,default=-100,help='The noise power in dBm')

        self.add_argument('--power_sleep', type=float, default=10, help='Power of BS in sleep mode Watts')
        self.add_argument('--power_active', type=float, default=50, help='Power of BS in active mode Watts')
        self.add_argument('--max_transm_power',type=float,default=35,help='Maximum transmission power in dbm')
        self.add_argument('--min_transm_power',type=float,default=11,help='Maximum transmission power in dBm')

        self.add_argument('--max_power',type=float,default=100,help='Maximum power of bS')
        self.add_argument('--min_power',type=float,default=20,help='Minimum power of BS')
        self.add_argument('--band', type=float, default=10.e6, help='the bandwidth Hz')

        self.add_argument('--min_datarate', type=float, default=5, help='the minimum data rate required in mb')
        self.add_argument('--max_datarate', type=float, default=20, help='the minimum data rate required in mb')

        self.add_argument('--avg_datarate_min', type=float, default=200, help='the minimum data rate served from the cell')
        self.add_argument('--avg_datarate_max', type=float, default=400, help='the maximum data rate served from the cell')

        self.add_argument('--avg_pwr_min', type=float, default=200, help='the minimum average power served from the cell')
        self.add_argument('--avg_pwr_max', type=float, default=400, help='the maximum average power served from the cell')

        # self.add_argument('--tm', type=float, default=1, help='tm SINR Gap')
        self.add_argument('--min_sinr', type=float, default=0.3, help='Minimum SINR')
        self.add_argument('--max_sinr', type=float, default=20, help='MAximum SINR')
      


    def __init_dqn(self):
        self.add_argument('--lr', type=float, default=1.e-3, help='learning rate for dqn')

    def __init_agent(self):
        self.add_argument('--observations', type=int, default=100, help='observations steps')
        self.add_argument('--update', type=int, default=8, help='n step q learning')
        self.add_argument('--tests', type=int, default=10, help='testing episode')
        self.add_argument('--episodes', type=int, default=10, help='training episode')
        self.add_argument('--epsilon_steps', type=int, default=4000, help='episodes for epsilon greedy explore')
        self.add_argument('--epochs', type=int, default=2, help='training epochs for each episode')

        self.add_argument('--save_ep', type=int, default=20, help='save model every n episodes')
        self.add_argument('--load_id', type=str, default=None, help='the model id to restore')

        self.add_argument('--gamma', type=float, default=0.99, help='reward discount rate')

        self.add_argument('--buffer_size', type=int, default=100000, help='size of replay buffer')
        self.add_argument('--mini_batch', type=int, default=64, help='size of mini batch')

        self.add_argument('--epsilon_init', type=float, default=0.9, help='initial value of explorer epsilon')
        self.add_argument('--epsilon_final', type=float, default=0.01, help='final value of explorer epsilon')

    def __init_env(self):
        self.add_argument('--random_seed', type=int, default=10000, help='seed of random generation')
        # self.add_argument('--dir_sum', type=str, default=self.__home_out('sum'), help='the path of tf summary')
        # self.add_argument('--dir_mod', type=str, default=self.__home_out('mod'), help='the path of tf module')
        # self.add_argument('--dir_log', type=str, default=self.__home_out('log'), help='the path of tf log')
        self.add_argument('--run_id', type=str, default=self.TIME_STAMP)

        
        