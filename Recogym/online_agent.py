import gym, recogym

# env_0_args is a dictionary of default parameters (i.e. number of products)
from recogym import env_1_args, Configuration

# You can overwrite environment arguments here:
env_1_args['random_seed'] = 42

# Initialize the gym for the first time by calling .make() and .init_gym()
env = gym.make('reco-gym-v1')
env.init_gym(env_1_args)

# Create list of hard coded actions.
actions = [None] + [1, 2, 3, 4, 5]

# Reset env and set done to False.
env.reset()
done = False

# Counting how many steps.
i = 0

while not done and i < len(actions):
    action = actions[i]
    observation, reward, done, info = env.step(action)
    print(f"Step: {i} - Action: {action} - Observation: {observation.sessions()} - Reward: {reward}")
    i += 1