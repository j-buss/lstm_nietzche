#Load Libraries
import numpy as np
import keras
from keras import layers
import random
import sys
import datetime
import time
import csv
import os
import logging
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('utils.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class Nietzche_Data():
    """Define the class for obtaining and preparing Nietzche text data."""
    def __init__(self, step=3, maxlen=60):
        self.step = step
        self.maxlen = maxlen
        self.text = self.get_data()
        self.chars = self.get_chars()
        self.char_indices = self.get_char_indices()
        self.x, self.y, self.len_sentences = self.prepare_x_y()

    def get_data(self):
      """Function to obtain the Nietzche text"""
      path = keras.utils.get_file(
          'nietzsche.txt',
          origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
      text = open(path).read().lower()
      #Adding in the following 2 lines to clean up the text and make it more uniform
      text = text.replace('\n', ' ')
      text = re.sub("[^a-zA-Z ]+", "", text)
      return text

    def get_chars(self):
      """Generate the list of unique characters in the corpus"""
      chars = sorted(list(set(self.text)))
      return chars

    def get_char_indices(self):
      """Generate a dictionary of unique characters in the corpus"""
      char_indices = dict((char, self.chars.index(char)) for char in self.chars)
      return char_indices

    def prepare_x_y(self):
      """Prepare the x and y datasets for training and execution"""
      # This holds our extracted sequences
      sentences = []

      # This holds the targets (the follow-up characters)
      next_chars = []

      for i in range(0, len(self.text) - self.maxlen, self.step):
        sentences.append(self.text[i: i + self.maxlen])
        next_chars.append(self.text[i + self.maxlen])
      x = np.zeros((len(sentences), self.maxlen, len(self.chars)), dtype=np.bool)
      y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)
      for i, sentence in enumerate(sentences):
          for t, char in enumerate(sentence):
              x[i, t, self.char_indices[char]] = 1
          y[i, self.char_indices[next_chars[i]]] = 1
      return x, y, len(sentences)

    def return_data(self):
      return self.x, self.y

def get_seed_text(text, maxlen):
  """Return a text seed at random from original corpus"""
  start_index = random.randint(0, len(text) - maxlen - 1)
  seed_text = text[start_index: start_index + maxlen]
  return seed_text

def sl_lstm_model(chars, maxlen):
  model = keras.models.Sequential()
  model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
  model.add(layers.Dense(len(chars), activation='softmax'))
  optimizer = keras.optimizers.RMSprop(lr=0.01)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer)
  return model

def sample(preds, temperature=1.0):
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def generate_text(model, maxlen, chars, char_indices, seed_text, temperature, create_str_len):
    generated_text=seed_text
    for i in range(create_str_len):
        sampled = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(generated_text):
            sampled[0, t, char_indices[char]] = 1.
        preds = model.predict(sampled, verbose = 0)[0]
        next_index = sample(preds, temperature)
        next_char = chars[next_index]
        generated_text += next_char
        generated_text = generated_text[1:]
    return generated_text

def nice_mk_dir(directory_path):
  try:
    os.stat(directory_path)
  except:
    os.mkdir(directory_path)
