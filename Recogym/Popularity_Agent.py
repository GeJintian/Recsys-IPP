import numpy as np
from numpy.random import choice
from recogym.agents import Agent
import gym, recogym

# env_0_args is a dictionary of default parameters (i.e. number of products)
from recogym import env_1_args, Configuration

# Define an Agent class.
class PopularityAgent(Agent):
    def __init__(self, config):
        # Set number of products as an attribute of the Agent.
        super(PopularityAgent, self).__init__(config)

        # Track number of times each item viewed in Organic session.
        self.organic_views = np.zeros(self.config.num_products)

    def train(self, observation, action, reward, done):
        """Train method learns from a tuple of data.
            this method can be called for offline or online learning"""

        # Adding organic session to organic view counts.
        if observation:
            for session in observation.sessions():
                self.organic_views[session['v']] += 1

    def act(self, observation, reward, done):
        """Act method returns an action based on current observation and past
            history"""

        # Choosing action randomly in proportion with number of views.
        prob = self.organic_views / sum(self.organic_views)
        action = choice(self.config.num_products, p = prob)

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': prob[action]
            }
        }
#//////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////


# You can overwrite environment arguments here:
env_1_args['random_seed'] = 42

# Initialize the gym for the first time by calling .make() and .init_gym()
env = gym.make('reco-gym-v1')
env.init_gym(env_1_args)

# Instantiate instance of PopularityAgent class.
num_products = 10
agent = PopularityAgent(Configuration({
    **env_1_args,
    'num_products': num_products,
}))

# Resets random seed back to 42, or whatever we set it to in env_0_args.
env.reset_random_seed()

# Train on 1000 users offline.
num_offline_users = 1000

for _ in range(num_offline_users):

    # Reset env and set done to False.
    env.reset()
    done = False

    observation, reward, done = None, 0, False
    while not done:
        old_observation = observation
        action, observation, reward, done, info = env.step_offline(observation, reward, done)
        agent.train(old_observation, action, reward, done)

# Train on 100 users online and track click through rate.
num_online_users = 100
num_clicks, num_events = 0, 0

for _ in range(num_online_users):

    # Reset env and set done to False.
    env.reset()
    observation, _, done, _ = env.step(None)
    reward = None
    done = None
    while not done:
        action = agent.act(observation, reward, done)
        observation, reward, done, info = env.step(action['a'])

        # Used for calculating click through rate.
        num_clicks += 1 if reward == 1 and reward is not None else 0
        num_events += 1

ctr = num_clicks / num_events

print(f"Click Through Rate: {ctr:.4f}")