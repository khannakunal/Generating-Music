import numpy as np

class DataHandler:

    def __init__(self, batch_size, sequence_length, file_path):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.file_path = file_path
        self.file_content = ''
        self.char_to_index = {}
        self.index_to_char = {}
        self.__build_mappings()

    def get_vocab_size(self):
        return len(self.char_to_index)
    
    def get_next_batch(self):
        total_length = len(self.file_content)
        next_sample_index_multiplier = int(total_length / self.batch_size)
        
        for start in range(0, next_sample_index_multiplier - self.sequence_length, self.sequence_length):
            x_batched = np.zeros((self.batch_size, self.sequence_length))
            y_batched = np.zeros((self.batch_size, self.sequence_length, self.get_vocab_size()))
            
            for i in range(0, self.batch_size):
                for j in range(0, self.sequence_length):
                    x_batched[i, j] = self.numerical_data[i * next_sample_index_multiplier + start + j]
                    y_batched[i, j, self.numerical_data[i * next_sample_index_multiplier + start + j + 1]] = 1
                    
            yield  x_batched, y_batched
    
    def __build_mappings(self):
        self.file_content = open(self.file_path, 'r').read()
        character_set = set(self.file_content)
        curr_index = 0
        for character in character_set:
            self.char_to_index[character] = curr_index
            self.index_to_char[curr_index] = character
            curr_index += 1

		self.numerical_data = np.asarray([self.char_to_index[character] for character in self.file_content], dtype=np.int32)	