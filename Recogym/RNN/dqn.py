# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model
from keras.layers import Dense,Activation,LSTM,Input,Add, Embedding
from keras.optimizers import Adam,RMSprop
import gym, recogym
from recogym import env_l_args, Configuration
from recogym.agents import Agent



EPISODES = 1000

class DQNAgent(Agent):
    def __init__(self, config):
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

        #Build embedding layers
        input1 = Input(batch_shape = (None,None,self.input_dim + 3))#sample size, timestamps, dim
        embedded_layer = Embedding(output_dim = 25)(input1)

        #Build LSTM. Two layers        
        Bandit_lstm_layer1, self.bandit_hidden1, self.bandit_cell1 = LSTM(25,return_sequences = True)(embedded_layer)
        Bandit_lstm_layer2, self.bandit_hidden2, self.bandit_cell2 = LSTM(10)(Bandit_lstm_layer1)

        # Neural Net for Deep-Q learning Model
        DQN_layer1 = Dense(24, activation='relu')(Bandit_lstm_layer2)
        DQN_layer2 = Dense(24, activation='relu')(DQN_layer1)
        DQN_layer3 = Dense(self.action_size, activation='linear')(DQN_layer2)
        model = Model(input1, DQN_layer3)
        model.compile(optimizer = 'adam', loss = 'mse')
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
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
    



if __name__ == "__main__":
    # You can overwrite environment arguments here:
    env_l_args['random_seed'] = 42

    # Initialize the gym for the first time by calling .make() and .init_gym()
    env = gym.make('reco-gym-v1')
    env.init_gym(env_l_args)
    env.reset_random_seed()

    #Change product number here
    num_products = 500
    agent = DQNAgent(Configuration({
        **env_l_args,
        'num_products': num_products,
    }))
    batch_size = 32
    env.reset()
    num_offline_users = 100000

    for _ in range(num_offline_users):
        # Reset env and set done to False.
        env.reset()
        done = False

        observation, reward, done = None, 0, False
        while not done:
            old_observation = observation
            action, observation, reward, done, info = env.step_offline(observation, reward, done)
            agent.train(old_observation, action, reward, done)





    #env = gym.make('CartPole-v1')
    #state_size = env.observation_space.shape[0]
    #action_size = env.action_space.n
    # agent.load("./save/cartpole-dqn.h5")


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