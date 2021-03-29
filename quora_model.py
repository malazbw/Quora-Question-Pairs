"""
@author: malazbw
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import sys
import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import pickle

def exponent_neg_manhattan_distance( left, right):

  return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))
class QuoraModel:

   # default constructor
  def __init__(self):
    
      self.tokenizer = pickle.load( open( "tokenizer.p", "rb" ) )
      self.g = tf.Graph()
      with self.g.as_default():
          self.model = tf.keras.models.load_model('model.h5', custom_objects={"exponent_neg_manhattan_distance": exponent_neg_manhattan_distance})
          print("loaaading")

  def text_to_wordlist(self,text, remove_stopwords=False, stem_words=False):
      # Clean the text, with the option to remove stopwords and to stem words.
      
      # Convert words to lower case and split them
      text = text.lower().split()

      # Optionally, remove stop words
      if remove_stopwords:
          stop_words = set(stopwords.words("english"))
          text = [w for w in text if not w in stop_words]
      
      text = " ".join(text)
      
      # Remove punctuation from text
      # text = "".join([c for c in text if c not in punctuation])

      # Clean the text
      text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
      text = re.sub(r"what's", "what is ", text)
      text = re.sub(r"\'s", " ", text)
      text = re.sub(r"\'ve", " have ", text)
      text = re.sub(r"can't", "cannot ", text)
      text = re.sub(r"n't", " not ", text)
      text = re.sub(r"i'm", "i am ", text)
      text = re.sub(r"\'re", " are ", text)
      text = re.sub(r"\'d", " would ", text)
      text = re.sub(r"\'ll", " will ", text)
      text = re.sub(r",", " ", text)
      text = re.sub(r"\.", " ", text)
      text = re.sub(r"!", " ! ", text)
      text = re.sub(r"\/", " ", text)
      text = re.sub(r"\^", " ^ ", text)
      text = re.sub(r"\+", " + ", text)
      text = re.sub(r"\-", " - ", text)
      text = re.sub(r"\=", " = ", text)
      text = re.sub(r"'", " ", text)
      text = re.sub(r":", " : ", text)
      text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
      text = re.sub(r" e g ", " eg ", text)
      text = re.sub(r" b g ", " bg ", text)
      text = re.sub(r" u s ", " american ", text)
      # text = re.sub(r"\0s", "0", text) # It doesn't make sense to me
      text = re.sub(r" 9 11 ", "911", text)
      text = re.sub(r"e - mail", "email", text)
      text = re.sub(r"j k", "jk", text)
      text = re.sub(r"\s{2,}", " ", text)
      

      
      # Return a list of words
      return(text)

  def preprocess_prepare(self, q1, q2):

      q1 = self.text_to_wordlist(q1)
      q2 = self.text_to_wordlist(q2)
      q1 = self.tokenizer.texts_to_sequences([q1])
      q2 = self.tokenizer.texts_to_sequences([q2])
      q1 = pad_sequences(q1, int(60))
      q2 = pad_sequences(q2, int(60))

      return q1,q2


  def predict(self, q1, q2):
    with self.g.as_default():

      q1,q2 = self.preprocess_prepare(q1, q2)
      sim = self.model.predict([np.array(q1),np.array(q2)])
      
      if(sim>0.5 ):
        return f"Similar ({sim})"
      else:
        return f"Not Similar ({sim})"


