from keras.layers import Input, Dense 
from keras.models import Model
import numpy as np

data = None
with open('data.npy', 'r') as f:
    data = np.load(f)
print 'loaded data!'

data = data[:,:,:,0]
print data.shape
train_data = data.reshape(len(data), np.prod(data.shape[1:]))
print train_data.shape[1]

#Sets the size of the hidden layer
encoding_dim = 10000

input_img = Input(shape=(train_data.shape[1],))
#first_layer = Dense(1000, activation='relu')(input_img)
encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
decoded = Dense(train_data.shape[1], activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='nadam', loss='binary_crossentropy')

print 'autoencoder created'

autoencoder.fit(train_data, train_data, epochs=10, batch_size=10, shuffle=True, verbose=2)
autoencoder.save('autoencoder.h5')
print 'saved autoencoder to autoencoder.h5'

encoder.save('encoder.h5')
print 'saved encoder to encoder.py'

decoder.save('decoder.h5')
print 'saved decoder to decoder.py'
