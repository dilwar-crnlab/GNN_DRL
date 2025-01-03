from typing import Tuple

import gym
import numpy as np

from .rmsa_env import RMSAEnv

from .optical_network_env import OpticalNetworkEnv

class DeepRMSAEnv(RMSAEnv):
    def __init__(
        self,
        num_bands,
        topology=None,
        j=1,
        episode_length=1000,
        mean_service_holding_time=25.0,
        mean_service_inter_arrival_time=0.1,
        node_request_probabilities=None,
        seed=None,
        k_paths=5,
        allow_rejection=False,
    ):
        super().__init__(
            num_bands=num_bands,
            topology=topology,
            episode_length=episode_length,
            load=mean_service_holding_time / mean_service_inter_arrival_time,
            mean_service_holding_time=mean_service_holding_time,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            k_paths=k_paths,
            allow_rejection=allow_rejection,
            reset=False,
        )

        self.j = j
        shape = 1 + 2 * self.topology.number_of_nodes() + (2 * self.j + 3) * self.k_paths * self.num_bands
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.uint8, shape=(shape,))
        self.action_space = gym.spaces.Discrete(self.k_paths  * self.num_bands * self.j + self.reject_action)
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.reset(only_episode_counters=False)

    def step(self, action: int):
        parent_step_result = None
        valid_action = False

        if action < self.k_paths * self.j * self.num_bands:  # action is for assigning a route
            valid_action = True
            route, band, block = self._get_route_block_id(action)

            initial_indices, lengths = self.get_available_blocks(route, self.num_bands, band, self.modulations)
            slots = self.get_number_slots(self.k_shortest_paths[self.current_service.source, self.current_service.destination][route], self.num_bands, band, self.modulations)
            if block < len(initial_indices):
                parent_step_result = super().step(
                    [route, band, initial_indices[block]])
            else:
                parent_step_result = super().step(
                    [self.k_paths, self.num_bands, self.num_spectrum_resources])
        else:
            parent_step_result = super().step(
                [self.k_paths, self.num_bands, self.num_spectrum_resources])

        obs, rw, _, info = parent_step_result
        info['slots'] = slots if valid_action else -1
        return parent_step_result

    def observation(self):
        # observation space defined as in https://github.com/xiaoliangchenUCD/DeepRMSCA/blob/eb2f2442acc25574e9efb4104ea245e9e05d9821/DeepRMSCA_Agent.py#L384
        source_destination_tau = np.zeros((2, self.topology.number_of_nodes()))
        min_node = min(self.current_service.source_id, self.current_service.destination_id)
        max_node = max(self.current_service.source_id, self.current_service.destination_id)
        source_destination_tau[0, min_node] = 1
        source_destination_tau[1, max_node] = 1
        spectrum_obs = np.full((self.k_paths * self.num_bands, 2 * self.j + 3), fill_value=-1.)
        # for the k-path ranges all possible bands to take the best decision
        for idp, path in enumerate(self.k_shortest_paths[self.current_service.source, self.current_service.destination]):
          for band in range(self.num_bands):
            available_slots = self.get_available_slots(path, band)
            num_slots = self.get_number_slots(path, self.num_bands, band, self.modulations)
            initial_indices, lengths = self.get_available_blocks(idp, self.num_bands, band, self.modulations)
            for idb, (initial_index, length) in enumerate(zip(initial_indices, lengths)):
                        # initial slot index
                spectrum_obs[idp + (self.k_paths * band), idb * 2 + 0] = 2 * (initial_index - .5 * self.num_spectrum_resources) / self.num_spectrum_resources

                        # number of contiguous FS available
                spectrum_obs[idp + (self.k_paths * band), idb * 2 + 1] = (length - 8) / 8
            spectrum_obs[idp + (self.k_paths * band), self.j * 2] = (num_slots - 5.5) / 3.5 # number of FSs necessary

            idx, values, lengths = DeepRMSAEnv.rle(available_slots)

            av_indices = np.argwhere(values == 1) # getting indices which have value 1
            # spectrum_obs = matrix with shape k_routes x s_bands in the scenario
            spectrum_obs[idp + (self.k_paths * band), self.j * 2 + 1] = 2 * (np.sum(available_slots) - .5 * self.num_spectrum_resources) / self.num_spectrum_resources # total number of available FSs
            spectrum_obs[idp + (self.k_paths * band), self.j * 2 + 2] = (np.mean(lengths[av_indices]) - 4) / 4 # avg. number of FS blocks available
        bit_rate_obs = np.zeros((1, 1))
        bit_rate_obs[0, 0] = self.current_service.bit_rate / 100

        return np.concatenate((bit_rate_obs, source_destination_tau.reshape((1, np.prod(source_destination_tau.shape))),
                               spectrum_obs.reshape((1, np.prod(spectrum_obs.shape)))), axis=1)\
            .reshape(self.observation_space.shape)


    def reward(self, band, path_selected):
        return 1 if self.current_service.accepted else -1

    def reset(self, only_episode_counters=True):
        return super().reset(only_episode_counters=only_episode_counters)

    def _get_route_block_id(self, action: int) -> Tuple[int, int]:
        route = action // (self.j * self.num_bands)
        band  = action // (self.j * self.k_paths)
        block = action % self.j
        return route, band, block


def shortest_path_first_fit(env: DeepRMSAEnv) -> int:
    if not env.allow_rejection:
        return 0
    else:
        initial_indices, _ = env.get_available_blocks(0)
        if len(initial_indices) > 0:  # if there are available slots
            return 0
        else:
            return env.k_paths * env.j


def shortest_available_path_first_fit(env: DeepRMSAEnv) -> int:
    for idp, _ in enumerate(
        env.k_shortest_paths[
            env.current_service.source, env.current_service.destination
        ]
    ):
        initial_indices, _ = env.get_available_blocks(idp)
        if len(initial_indices) > 0:  # if there are available slots
            return idp * env.j  # this path uses the first one
    return env.k_paths * env.j
