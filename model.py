import os
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding

log_directory = '.\\model'

def save_model_weights(epoch, model):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    model.save_weights(os.path.join(log_directory, 'weights.{}'.format(epoch)))
	
def build_model(vocab_size, batch_size, sequence_length):
	model = Sequential()
	model.add(Embedding(vocab_size, 512, batch_input_shape = (batch_size, sequence_length)))
	for i in range(3):
		model.add(LSTM(256, return_sequences = True, stateful = True))
		model.add(Dropout(0.2))
	model.add(TimeDistributed(Dense(vocab_size))) 
	model.add(Activation('softmax'))
	return model
		
	