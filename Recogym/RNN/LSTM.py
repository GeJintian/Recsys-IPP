import numpy as np
np.random.seed(1337)  # for reproducibility
import random
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Activation,LSTM,Input,Add, Embedding
from keras.optimizers import RMSprop



class State_processor:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.session = [1,0]#first stands for Organic, second stands for Session
        

    
    def build_network(self):

        #Build embedding layers
        input1 = Input(batch_shape = (None,None,self.input_dim))#sample size, timestamps, dim
        embedded_layer = Embedding(output_dim = 25)(input1)

        #Build Organic LSTM. Two layers        
        Organic_lstm_layer1, self.organic_hidden1, self.organic_cell1 = LSTM(25,stateful = True,return_state=True,return_sequences = True)(embedded_layer)
        Organic_lstm_layer2, self.organic_hidden2, self.organic_cell2 = LSTM(self.output_dim,stateful = True,return_sequences = True,
                                                                             return_state = True)(Organic_lstm_layer1)


        # Build Bandit LSTM. Two layers
        Bandit_lstm_layer1, self.bandit_hidden1, self.bandit_cell1 = LSTM(25, stateful=True,return_sequences = True, return_state=True)(embedded_layer)
        Bandit_lstm_layer2, self.bandit_hidden2, self.bandit_cell2 = LSTM(self.output_dim, stateful=True,return_sequences = True,
                                                                             return_state=True)(Bandit_lstm_layer1)
        Total_layer = Add()([Bandit_lstm_layer2 * self.session[1], Organic_lstm_layer2 * self.session[0]])
        self.LSTM = Model(input1,Total_layer)
        self.LSTM.compile(optimizer='adam', loss='mse')

    #Transfer states from Organic session
    "No need to change the following part"
    def Organic_2_Bandit(self):
        self.LSTM.layers[1].state[0] = self.organic_hidden1
        self.LSTM.layers[1].state[1] = self.organic_cell1
        self.LSTM.layers[2].state[0] = self.organic_hidden2
        self.LSTM.layers[2].state[1] = self.organic_cell2

    def Bandit_2_Organic(self):
        self.LSTM.layers[1].state[0] = self.bandit_hidden1
        self.LSTM.layers[1].state[1] = self.bandit_cell1
        self.LSTM.layers[2].state[0] = self.bandit_hidden2
        self.LSTM.layers[2].state[1] = self.bandit_cell2

