#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import ast
import random
import re
import keras
import random
from collections import OrderedDict
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import Counter, defaultdict
from operator import itemgetter
import matplotlib.pyplot as plt


# ## READING INPUT FILE

# In[2]:


filepath = "data/mimic3_data.pkl"


# In[3]:


with open(filepath, "rb") as new_file:
    df = pickle.load(new_file)


# In[4]:


df.head()


# ## Pre processing raw text and ICD 9 codes

# In[7]:


def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=101):
    """Splits the input and labels into 3 sets"""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_size+test_size), random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(val_size+test_size), random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def replace_with_grandparent_codes(string_codes, ICD9_CODE_FIRST_LEVEL):
    """replace_with_grandparent_codes takes a list of ICD9_CODE codes and 
    returns the list of their grandparents ICD9 code in the first level of the ICD9 hierarchy"""
    ICD9_CODE_RANGES = [x.split('-') for x in ICD9_CODE_FIRST_LEVEL]
    resulting_codes = []
    for code in string_codes:
        for i, gparent_range in enumerate(ICD9_CODE_RANGES):
            range = int(gparent_range[1]) if len(gparent_range) == 2 else int(gparent_range[0])
            if code[0]!='E' and code[0]!='V':
                if int(code[0:3]) <= range:
                    resulting_codes.append(ICD9_CODE_FIRST_LEVEL[i])
                    break
    return list(set(resulting_codes))

# Get HoPI document 
def get_HoPI(text):
    str_list = re.findall("history.*\n\n\n\n", text, re.DOTALL)
    if len(str_list) == 0:
        str_list = re.findall("history.*\n\n\n", text, re.DOTALL)
    if len(str_list) > 0:
        return "".join(str_list)
    else:
        return ""

# Process special sequences
def clean_special_seq(text):
    # Replacing dates with space character
    text = re.sub("\[\*\*\d{4}-.*?\*\*\]"," ", text)
    # Finding special sequences
    string = re.findall("\[\*\*.*?\*\*\]", text)
    # Replacing special sequences with firs occurence
    clean_text = text
    for each_string in string:
        temp_str_list = each_string[3:-3].split(" ")
        clean_text = clean_text.replace(each_string, temp_str_list[0])
    return clean_text

# Handling special characters and numbers
def clean_char(text):
    result = text.replace("\n", " ")
    """ Canonize numbers"""
    result = re.sub(r"(\d+)", "DG", result)
    return result

# Complete processing of text
def process_text(list_of_text):
    processed_text = []
    for text in list_of_text:
        text = text.lower()
        # Filtering the HoPI document
        result = get_HoPI(text)
        if result != "":
            # process special sequences
            result = clean_special_seq(result)
            # clean new line characters and numbers
            result = clean_char(result)
        processed_text.append(result)
    return processed_text


# In[8]:


# Vectorize ICD codes

def vectorize_icd_string(x, code_list):
    """Takes a string with ICD codes and returns an array of the right of 0/1"""
    r = []
    for code in code_list:
        if code in x: r.append(1)
        else: r.append(0)
    return np.asarray(r)

def vectorize_icd_labels(labels, code_list):
    """Takes list of labels and converts it into a vector of 0/1"""
    vector_icd = []
    for x in labels:
        vector_icd.append(vectorize_icd_string(x, code_list))
    return np.asarray(vector_icd, dtype=np.float32)

# Vectorize and Pad notes Text

def vectorize_notes(text, MAX_NB_WORDS=None, OOV_token=None, word_frequency=2,
                    verbose = True):
    """Takes a note column and encodes it into a series of integer
        Also returns the dictionnary mapping the word to the integer
        and tokenizer json str for creating tokenizer from it"""
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token=OOV_token)
    tokenizer.fit_on_texts(text)
    config = tokenizer.get_config()

    word_count = ast.literal_eval(config['word_counts'])
    new_word_count = OrderedDict()
    for word in word_count.keys():
        if word_count[word] >= 5:
            new_word_count[word] = word_count[word]

    words = list(new_word_count.keys())
    random.shuffle(words)
    new_word_index = dict(zip(words, list(range(2, len(words)+1))))
    new_word_index[OOV_token] = 1
    new_index_word = {v:k for k,v in new_word_index.items()}

    tokenizer.word_counts = new_word_count
    tokenizer.index_word = new_index_word
    tokenizer.word_index = new_word_index

    data = tokenizer.texts_to_sequences(text)
    note_length =  [len(x) for x in data]
    vocab = tokenizer.word_index
    MAX_VOCAB = len(vocab)
    if verbose:
        print('Vocabulary size: %s' % MAX_VOCAB)
        print('Average note length: %s' % np.mean(note_length))
        print('Max note length: %s' % np.max(note_length))
    
    tokenizer_json = tokenizer.to_json()
    return data, vocab, MAX_VOCAB, tokenizer_json

def pad_notes(data, MAX_SEQ_LENGTH):
    data = pad_sequences(data, maxlen=MAX_SEQ_LENGTH, value=0)
    return data, data.shape[1]


# Creates an embedding Matrix
# Based on https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

def embedding_matrix(f_name, dictionary, EMBEDDING_DIM, verbose = True, sigma = None):
    """Takes a pre-trained embedding and adapts it to the dictionary at hand
        Words not found will be all-zeros in the matrix"""

    # Dictionary of words from the pre trained embedding
    pretrained_dict = {}
    with open(f_name, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            pretrained_dict[word] = coefs

    # Default values for absent words
    if sigma:
        pretrained_matrix = sigma * np.random.rand(len(dictionary) + 1, EMBEDDING_DIM)
    else:
        pretrained_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))
    
    # Substitution of default values by pretrained values when applicable
    for word, i in dictionary.items():
        vector = pretrained_dict.get(word)
        if vector is not None:
            pretrained_matrix[i] = vector

    if verbose:
        print('Vocabulary in notes:', len(dictionary))
        print('Vocabulary in original embedding:', len(pretrained_dict))
        inter = list( set(dictionary.keys()) & set(pretrained_dict.keys()) )
        print('Vocabulary intersection:', len(inter))

    return pretrained_matrix, 


# In[9]:


ICD9_CODE_FIRST_LEVEL = ['001-139','140-239','240-279','290-319', '320-389', '390-459','460-519', '520-579', '580-629',
                         '630-679', '680-709','710-739', '760-779', '780-789', '790-796', '797', '798', '799', '800-999']


# In[10]:


ICD9_CODE_SECOND_LEVEL = ['001-009', '010-018', '020-027', '030-041', '042-042', '045-049', '050-059', '060-066',
                          '070-079', '080-088','090-099', '100-104', '110-118', '120-129','130-136', '137-139', '140-149',
                          '150-159', '160-165', '170-176', '179-189', '190-199', '200-209', '210-229', '230-234', '235-238',
                          '239-239', '240-246', '249-259', '260-269', '270-279', '280', '281', '282', '283', '284', '285', '286',
                          '287', '288', '289', '290-294', '295-299', '300-316', '317-319', '320-327', '330-337', '338-338', '339-339',
                          '340-349', '350-359', '360-379', '380-389', '390-392', '393-398', '401-405', '410-414', '415-417', '420-429',
                          '430-438', '440-449', '451-459', '460-466', '470-478', '480-488', '490-496', '500-508', '510-519', '520-529',
                          '530-539', '540-543', '550-553', '555-558', '560-569', '570-579','580-589', '590-599', '600-608', '610-612',
                          '614-616', '617-629', '630-639', '640-649', '650-659', '660-669', '670-677', '678-679', '680-686', '690-698',
                          '700-709', '710-719', '720-724', '725-729', '730-739', '740-742', '743-744', '745-747', '748-748', '749-751', '752-752',
                          '753-753', '754-756', '757-757', '758-758', '759-759', '760-763', '764-779', '780-789',
                          '790-796', '797-799', '800-804', '805-809', '810-819', '820-829', '830-839', '840-848', '850-854', '860-869',
                          '870-879', '880-887', '890-897', '900-904', '905-909', '910-919', '920-924', '925-929', '930-939', '940-949',
                          '950-957', '958-959', '960-979', '980-989', '990-995', '996-999']


# ## Train Test Split

# In[11]:


X = df['TEXT']
y = df['ICD9_CODE']
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)


# ## Process Training data

# In[12]:


X_train_p = process_text(X_train)


# In[13]:


level_1_labels = [replace_with_grandparent_codes(x, ICD9_CODE_FIRST_LEVEL) for x in y_train]


# In[14]:


level_2_labels = [replace_with_grandparent_codes(x, ICD9_CODE_SECOND_LEVEL) for x in y_train]


# In[15]:


train_df = pd.DataFrame(data={'TEXT':X_train_p, 'Level_1':level_1_labels, 'Level_2':level_2_labels}, index=None)


# In[16]:


train_df_clean = train_df[train_df['TEXT']!=""].copy().reset_index().drop('index', axis=1)


# In[17]:


X_train_final = np.asarray(train_df_clean['TEXT'].values)
level_1_final = np.asarray(train_df_clean['Level_1'].values)
level_2_final = np.asarray(train_df_clean['Level_2'].values)


# ## Tokenizer converting text to sequence 

# In[18]:


data, vocab, MAX_VOCAB, tokenizer_json = vectorize_notes(text=X_train_final,
                                                         OOV_token="<UNK>",
                                                         word_frequency=5)


# In[19]:


padded_data, MAX_SEQ_LENGTH = pad_notes(data=data, MAX_SEQ_LENGTH=500)


# ## Making embedding vectors for text

# In[20]:


embedding_dim = 200
input_length = 500
batch_size=256


# In[21]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(MAX_VOCAB+1, embedding_dim, input_length=input_length))

model.compile('rmsprop', 'mse')


# In[22]:


x_embedding_batch = model.predict(padded_data[0:batch_size])
print(x_embedding_batch.shape)


# ## Making embedding vectors for label representation

# In[23]:


level_1 = vectorize_icd_labels(level_1_final, ICD9_CODE_FIRST_LEVEL)
level_2 = vectorize_icd_labels(level_2_final, ICD9_CODE_SECOND_LEVEL)


# In[ ]:




