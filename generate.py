import random
import numpy as np
import os
import json
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Embedding

log_directory = '.\\model'
data_dir = '.\\data'

def load_weights(model):
    model.load_weights(os.path.join(log_directory, 'weights.99.h5'))
	
def build_sample_model(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 512, batch_input_shape=(1, 1)))
    for i in range(3):
        model.add(LSTM(256, return_sequences = (i != 2), stateful = True))
        model.add(Dropout(0.2))

    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    return model

def generate():
	with open(os.path.join(data_dir, 'char_to_index.json')) as f:
		char_to_index = json.load(f)

	index_to_char = {index: char for (char, index) in char_to_index.items()}

	vocab_size = len(index_to_char)

	model = build_sample_model(vocab_size)
	load_weights(model)
	
	random_char_index = random.randint(0, 86)
	generated_music = []
	chars_to_generate = 3000

	generated_music.append(random_char_index)

	next_char_index = random_char_index

	for i in range(chars_to_generate - 1):
		x_input = np.array([next_char_index]).reshape(1, 1)
		probability_vector = model.predict_on_batch(x_input).ravel()
		next_char_index = np.random.choice(np.arange(0, 86), p = probability_vector)
		generated_music.append(next_char_index)
	
	generated_music_list = [index_to_char[x] for x in generated_music]
	generated_music_str = ''.join(generated_music_list)	
	return generated_music_str

if __name__ == '__main__':
    generate()		