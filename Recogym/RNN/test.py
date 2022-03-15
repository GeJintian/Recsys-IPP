import random
import numpy as np
from collections import deque
from keras.models import Model,Sequential
from keras.layers import Dense,Activation,LSTM,Input,Add, Embedding
from keras.optimizers import Adam,RMSprop
import gym, recogym
from recogym import env_1_args, Configuration
from recogym.agents import Agent


model = Sequential()


#Neural Net for Deep-Q learning Model
model.add(Dense(100, input_dim = 500, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(500, activation='linear'))
model.compile(optimizer = 'adam', loss = 'mse')

a=np.zeros((1,500))
print(a.shape)
model.predict(a)