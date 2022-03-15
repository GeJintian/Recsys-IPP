from keras.layers import Dense, LSTM, Masking
from keras.models import Sequential
import numpy as np
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(5, 3)))
model.add(LSTM(5,return_sequences=True))

# Neural Net for Deep-Q learning Model
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.add(Dense(2, activation='linear'))
model.compile(optimizer = 'adam', loss = 'mse')

a = np.array([1,2,3,4,5,6,7,8,9,10,11,12,15,20,30]).reshape((1,5,3))
print(a)

output = model.predict(a)[0]
print(output,output.shape)
max = np.amax(output[3])
print(max)