from data_handler import DataHandler
from model import build_model, save_model_weights
import json
import os

batch_size = 16
sequence_length = 64
epochs = 100

data_dir = '.\\data'

def train(path):
	dh = DataHandler(batch_size, sequence_length, path)	
	
	with open(os.path.join(data_dir, 'char_to_index.json'), 'w') as f:
		json.dump(dh.char_to_index, f)		

	model = build_model(dh.get_vocab_size(), batch_size, sequence_length)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])	
	
	for epoch in range(epochs):
		step_count = 0
		for x_batched, y_batched in dh.get_next_batch():
			loss, accuracy = model.train_on_batch(x_batched, y_batched)
			
			if step_count % 10 == 0:
				print('After Epoch: {} and steps: {}, loss is: {} and accuracy is: {}'.format(epoch, step_count, loss, accuracy))	
				
			step_count += 1
	
		save_model_weights(epoch, model)	
		
if __name__ == '__main__':
    train('.\\data\\input.txt')