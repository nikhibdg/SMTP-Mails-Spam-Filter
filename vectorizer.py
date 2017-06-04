from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(__file__)



vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None) 
