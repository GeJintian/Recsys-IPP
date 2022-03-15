import random
import numpy as np
from collections import deque
from keras.models import Model,Sequential
from keras.layers import Dense,Activation,LSTM,Input,Add, Embedding,Masking
from keras.optimizers import Adam,RMSprop
import gym, recogym
from recogym import env_1_args, Configuration
from recogym.agents import Agent
import matplotlib.pyplot as pt


class DQNAgent(Agent):
    def __init__(self,config):
        
        # Set number of products as an attribute of the Agent.
        super(DQNAgent, self).__init__(config)

        self.input_dim = self.config.num_products
        self.action_size = self.config.num_products
        self.memory = deque(maxlen=200)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.times = 200
        self.n_layer = self.action_size * 3
        self.model = self._build_model()


    def _build_model(self):
        
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(self.times, self.input_dim + 1)))
        model.add(Dense(self.n_layer, activation='relu'))
        model.add(LSTM(self.n_layer,return_sequences = True))

        # Neural Net for Deep-Q learning Model
        model.add(Dense(self.n_layer, activation='relu'))
        model.add(Dense(self.n_layer,activation='sigmoid'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(optimizer = 'adam', loss = 'mse')
        return model


    def remember(self, states, actions, rewards, size):
        self.memory.append((states, actions, rewards, size))

    def act(self, states, count_step):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(states)
        return np.argmax(act_values[0][count_step])  # returns action

    def replay(self, batch_size):
        #print('S1')
        '''
        minibatch = random.sample(self.memory, batch_size)

        for states, action, reward, next_state, done in minibatch:
            #print('1')
            target = reward
            #print('2')
            if not done:
                #print('3')
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            #print('4')
            target_f = self.model.predict(states)
            #print('5')
            target_f[0][action] = target
            a = self.model.fit(states, target_f, epochs=1, verbose=0)
            b = abs(float(a.history['loss'][0]))
            self.Loss.append(b)
            #print('6')
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            '''
        minibatch = random.sample(self.memory, batch_size)
        for states, actions, rewards, size in minibatch:
            #print('1')
            target_q = self.model.predict(states)
            #print('3')
            for i in range(size - 1):
                if rewards[i] != -1:
                    target_q[0,i,actions[i]] = rewards[i] + self.gamma * np.amax(target_q[0, i+1])
            
            #print('4')
            a = self.model.fit(states, target_q, epochs=1, verbose=0)
            b = abs(float(a.history['loss'][0]))
            #print(b)
            #print('6')
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
    



# You can overwrite environment arguments here:
env_1_args['random_seed'] = 42
env_1_args['num_products'] = 50
# Initialize the gym for the first time by calling .make() and .init_gym()
env = gym.make('reco-gym-v1')
env.init_gym(env_1_args)
env.reset_random_seed()
num_products = 50


num_products = env_1_args['num_products']
agent = DQNAgent(Configuration({
    **env_1_args,
    'num_products': num_products,
}))
batch_size = 10

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
num_online_users = 200
num_clicks, num_events = 0,0
time_step = 200

count = 0
state0 = np.zeros((1,time_step,num_products+1))
actions0 = np.zeros(time_step, dtype=np.int32)
rewards0 = np.zeros(time_step)

    
for users in range(num_online_users):#Training part
# Reset env and set done to False.
    #print(count)
    count = count + 1
    env.reset()
    observation, _, done, _ = env.step(None)
    reward = None
    states = state0.copy()
    actions = actions0.copy()
    rewards = rewards0.copy()
    action = None
    count_step = -1
    ctrs = []

    while not done and count_step < time_step:
        if observation.current_sessions.__len__() > 0:
            for session in observation.sessions():
                if count_step + 1 < time_step:
                    count_step += 1
                    states[0, count_step, session['v']] = 1
                    states[0, count_step, num_products] = 1
                    rewards[count_step] = -1
                    actions[count_step] = num_products
                else:
                    break

        if count_step < time_step:
            action = agent.act(states, count_step)
            actions[count_step] = action
            observation, reward, done, info = env.step(action)
            rewards[count_step] = reward
            if action and observation.current_sessions.__len__() > 0:
                reward += 1
            count_step += 1
            if count_step < time_step:
                states[0,count_step,action] = 1 
                states[0,count_step,num_products] = 0 #Indicating Bandit session

        
        #print('s7')
        # Used for calculating click through rate.
    
        if True:#count>8000:
            num_clicks += 1 if reward > 0 else 0
            num_events += 1
    agent.remember(states, actions, rewards, count_step)

    if users + 1 >= batch_size and (users + 1) % batch_size == 0:
        agent.replay(batch_size)
        ctr = num_clicks / num_events
        ctrs.append(ctr)
        print(f"Click Through Rate: {ctr:.4f}")
        num_clicks = 0
        num_events = 0
        A=np.array(ctrs)
        B=np.array(range(len(A)))
        pt.plot(B,A)
        pt.show()




'''   
done = False
batch_size = 32
for e in range(EPISODES):
    states = env.reset()
    states = np.reshape(states, [1, state_size])
    for time in range(500):
        # env.render()
        action = agent.act(states)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(states, action, reward, next_state, done)
        states = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, EPISODES, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    # if e % 10 == 0:
    #     agent.save("./save/cartpole-dqn.h5")
    '''