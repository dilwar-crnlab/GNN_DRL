import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gym

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from IPython.display import clear_output

#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'


import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.policies import MlpPolicy


from stable_baselines3.common import results_plotter
stable_baselines3.__version__ # printing out stable_baselines version used


# import stable_baselines3
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.results_plotter import load_results, ts2xy
# from stable_baselines3 import DQN  # Change from PPO to DQN
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.dqn.policies import MlpPolicy  # Change to DQN policy
# from stable_baselines3.common import results_plotter





# callback from https://stable-baselines.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1, show_plot: bool=False):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.show_plot = show_plot

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:

        if self.show_plot and self.n_calls % self.check_freq == 0 and self.n_calls > 5001:
            plotting_average_window = 100

            training_data = pd.read_csv(self.log_dir + 'training.monitor.csv', skiprows=1)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 4.8))

            ax1.plot(np.convolve(training_data['r'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')

            ax2.semilogy(np.convolve(training_data['episode_service_blocking_rate'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Episode service blocking rate')

            ax3.semilogy(np.convolve(training_data['episode_bit_rate_blocking_rate'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Episode bit rate blocking rate')

            # fig.get_size_inches()
            plt.tight_layout()
            plt.show()

        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                 # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {} - ".format(self.num_timesteps), end="")
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                  # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)
                if self.verbose > 0:
                    clear_output(wait=True)

        return True






# loading the topology binary file containing the graph and the k-shortest paths
# if you want to generate your own binary topology file, check examples/create_topology_rmsa.py
topology_name = 'nsfnet_chen_link_span'
k_paths = 5
with open(f'../topologies/{topology_name}_{k_paths}-paths_6-modulations.h5', 'rb') as f:
    topology = pickle.load(f)

#monitor_info_keywords=('service_blocking_rate_since_reset','bit_rate_blocking_rate_since_reset')

monitor_info_keywords = (
            "service_blocking_rate",
            "episode_service_blocking_rate",
            "bit_rate_blocking_rate",
            "episode_bit_rate_blocking_rate",
)





# Scenario to run: 1=Band C, 2=Bands C+L, 3=Bands C+L+S, 4=Bands C+L+S+E
numbands=2
            
env_args = dict(num_bands=numbands, topology=topology, seed=10, #load is for traffic load
                allow_rejection=False, # the agent cannot proactively reject a request
                j=1, # consider only the first suitable spectrum block for the spectrum assignment
                mean_service_holding_time=10, # value is not set as in the paper to achieve comparable reward values
                mean_service_inter_arrival_time=0.1,
                k_paths=5,
                episode_length=100, node_request_probabilities=None)


topology.edges()


# Create log dir
#log_dir = "./tmp/deeprmsa-ppo/"
log_dir = "./tmp/deeprmsa-dqn/"  # Changed directory name to reflect DQN
os.makedirs(log_dir, exist_ok=True)
callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir, show_plot=False)

env = gym.make('DeepRMSA-v0', **env_args)

# logs will be saved in log_dir/training.monitor.csv
# in this case, on top of the usual monitored things, we also monitor service and bit rate blocking rates
env = Monitor(env, log_dir + 'training', info_keywords=monitor_info_keywords)
# for more information about the monitor, check https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/bench/monitor.html#Monitor

# here goes the arguments of the policy network to be used
policy_args = dict(net_arch=5*[128]) # we use the elu activation function



# Initialize the DQN agent with suitable hyperparameters
agent = PPO(MlpPolicy, env, verbose=0, tensorboard_log="./tb/PPO-DeepRMSA-v0/", policy_kwargs=policy_args, gamma=.95, learning_rate=10e-6)

#agent = PPO(MlpPolicy, env, verbose=0, tensorboard_log="./tb/PPO-DeepRMSA-v0/", policy_kwargs=policy_args, gamma=.95, learning_rate=10e-6)



a = agent.learn(total_timesteps=1000000, callback=callback)


results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "DeepRMSA DQN")


# first, we need to load the monitor data
training_data = pd.read_csv(log_dir + 'training.monitor.csv', skiprows=1)
training_data.describe()


plotting_average_window = 100

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 4.8))

ax1.plot(np.convolve(training_data['r'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')


# Overall service blocking rate
ax2.semilogy(np.convolve(training_data['service_blocking_rate'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))
ax2.set_xlabel('Episode')
ax2.set_ylabel('Service blocking rate')

# Overall bit rate blocking rate
ax3.semilogy(np.convolve(training_data['bit_rate_blocking_rate'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))
ax3.set_xlabel('Episode')
ax3.set_ylabel('Bit rate blocking rate')

ax2.semilogy(np.convolve(training_data['service_blocking_rate_since_reset'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'),
             color='red',  # Line color
            linewidth=0.8)

ax2.set_xlabel('Episode')
ax2.set_ylabel('Episode service blocking rate')

ax3.semilogy(np.convolve(training_data['bit_rate_blocking_rate_since_reset'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

ax3.set_xlabel('Episode')
ax3.set_ylabel('Episode bit rate blocking rate')

# fig.get_size_inches()
plt.tight_layout()
plt.show()
