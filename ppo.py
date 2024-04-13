import torch
from torch.distributions import MultivariateNormal
from network import FeedForwardNN

class PPO:
    def __init__(self, env):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self._init_hyperparameters()

        # initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

    def learn(self, total_timesteps):
        t_so_far = 0 # Timesteps simulated so far

        while t_so_far < total_timesteps:

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.99

    def rollout(self): # Collect experience by running the policy
        batch_obs = []          # batch observations
        batch_acts = []         # batch actions
        batch_log_probs = []    # log probs of each action
        batch_rews = []         # batch rewards
        batch_rtgs = []         # rewards-to-go
        batch_lens = []         # episodic lengths in batch

        obs = self.env.reset()
        done = False

        for ep_t in range(self.max_timesteps_per_episode):
            
