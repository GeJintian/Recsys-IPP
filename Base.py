import random
import numpy as np
from collections import deque
from keras.models import Model,Sequential
from keras.layers import Dense,Activation,LSTM,Input,Add, Embedding
from keras.optimizers import Adam,RMSprop
import gym, recogym
from recogym import env_1_args, Configuration
from recogym.agents import Agent


class DQNAgent(Agent):
    def __init__(self,config):
        
        # Set number of products as an attribute of the Agent.
        super(DQNAgent, self).__init__(config)

        self.input_dim = self.config.num_products
        self.action_size = self.config.num_products
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        
        model = Sequential()


        # Neural Net for Deep-Q learning Model
        model.add(Dense(100, input_dim = self.input_dim, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(optimizer = 'adam', loss = 'mse')
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state,[1,self.input_dim])
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        #print('S1')
        for state, action, reward, next_state, done in minibatch:
            #print('1')
            target = reward
            state = np.reshape(state,[1,self.input_dim])
            next_state = np.reshape(next_state,[1,self.input_dim])
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
    



# You can overwrite environment arguments here:
env_1_args['random_seed'] = 42
env_1_args['num_products'] = 500
# Initialize the gym for the first time by calling .make() and .init_gym()
env = gym.make('reco-gym-v1')
env.init_gym(env_1_args)
env.reset_random_seed()
num_products = 500

num_products = env_1_args['num_products']
agent = DQNAgent(Configuration({
    **env_1_args,
    'num_products': num_products,
}))
batch_size = 16

# .reset() env before each episode (one episode per user).
env.reset()
'''
count = 0
for _ in range(num_offline_users):
# Reset env and set done to False.
    print(count)
    count=count+1
    env.reset()
    done = False
    observation, reward, done = None, 0, False
    current_state = np.zeros(num_products)
    next_state = np.zeros(num_products)
    while not done:
        old_observation = observation
        action, observation, reward, done, info =env.step_offline(observation, reward, done)
        if observation:
            for session in observation.sessions():
                current_state[session['v']] += 1
        next_state = current_state
        if action:
            if reward == 0:
                next_state[action['a']] += -1
            if reward == 1:
                next_state[action['a']] += 1
            agent.remember(current_state, action['a'], reward, next_state, done)
        current_state = next_state
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
'''    
num_online_users = 10000
num_clicks, num_events = 0,0

count = 0
for _ in range(num_online_users):
# Reset env and set done to False.
    print(count)
    count = count + 1
    env.reset()
    observation, _, done, _ = env.step(None)
    reward = None
    done = None
    current_state = np.zeros(num_products)
    next_state = np.zeros(num_products)
    action = None
    if observation:
        for session in observation.sessions():
            current_state[session['v']] += 1
    while not done:
        if observation and not action:
            for session in observation.sessions():
                next_state[session['v']] += 1
        if action:
            if reward == 0:
                next_state[action] += -1
            if reward == 1:
                next_state[action] += 1
            agent.remember(current_state, action, reward, next_state, done)
            print('replying')
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        current_state = next_state
        action = agent.act(current_state)
        observation, reward, done, info = env.step(action)
        # Used for calculating click through rate.
        if count>8000:
            num_clicks += 1 if reward == 1 and reward is not None else 0
            num_events += 1


ctr = num_clicks / num_events
print(f"Click Through Rate: {ctr:.4f}")


'''   
done = False
batch_size = 32
for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, EPISODES, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    # if e % 10 == 0:
    #     agent.save("./save/cartpole-dqn.h5")
    '''