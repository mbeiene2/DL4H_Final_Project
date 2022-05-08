#!/usr/bin/env python
# coding: utf-8
# Michael Beiene
# DL4H UIUC
# Level 1 Labels model for GRU(X) - GRU(Z) and tfidf(d)-atomic
# May 2022
# Version 1.0



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
import time
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import *
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline





get_ipython().run_line_magic('load_ext', 'line_profiler')
get_ipython().run_line_magic('load_ext', 'memory_profiler')





seed = 1873
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# ## READING INPUT FILE




filepath = "data/mimic3_data.pkl"





with open(filepath, "rb") as new_file:
    df = pickle.load(new_file)





df.head()


# ## Pre processing raw text and ICD 9 codes




def replace_with_grandparent_codes(string_codes, ICD9_CODE_FIRST_LEVEL, add_start_end=True):
    """replace_with_grandparent_codes takes a list of ICD9_CODE codes and 
    returns the list of their grandparents ICD9 code in the first level of the ICD9 hierarchy"""
    ICD9_CODE_RANGES = [x.split('-') for x in ICD9_CODE_FIRST_LEVEL]
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.    
    resulting_codes = []
    for code in string_codes:
        for i, gparent_range in enumerate(ICD9_CODE_RANGES):
            range = int(gparent_range[1]) if len(gparent_range) == 2 else int(gparent_range[0])
            if code[0]!='E' and code[0]!='V':
                if int(code[0:3]) <= range:
                    resulting_codes.append(ICD9_CODE_FIRST_LEVEL[i])
                    break
    result = ' '.join(set(resulting_codes))
    if add_start_end:
        return '<start> ' + result + ' <end>' 
    else:
        return result

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
def process_text(list_of_text, add_start_end=True):
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
            result = result.strip()
            # adding a start and an end token to the sentence
            # so that the model know when to start and stop predicting.
            if add_start_end:
                result = '<start> ' + result + ' <end>'
        processed_text.append(result)
    return processed_text





# Vectorize ICD codes
def vectorize_icd_string(x, code_list):
    """Takes a string with ICD codes and returns an array of the right of 0/1"""
    r = []
    for code in code_list:
        if code in x: r.append(1)
        else: r.append(0)
    return np.asarray(r)

def vectorize_icd_labels(labels, code_list, drop_single_class_codes=False):
    """Takes list of labels and converts it into a vector of 0/1"""
    vector_icd = []
    for x in labels:
        vector_icd.append(vectorize_icd_string(x, code_list))
    vectors = np.asarray(vector_icd, dtype=np.float32)
    
    def get_multiclass_codes(vectors, code_list):
        multi_class_codes = []
        assert vectors.shape[1] == len(code_list)
        for i in range(vectors.shape[1]):
            if vectors[:, i].sum() > 0:
                multi_class_codes.append(code_list[i])
        return multi_class_codes
    
    if drop_single_class_codes:
        multiclass_codes = get_multiclass_codes(vectors, code_list)
        if len(multiclass_codes) != len(code_list):
            vector_icd = []
            for x in labels:
                vector_icd.append(vectorize_icd_string(x, multiclass_codes))
            vectors = np.asarray(vector_icd, dtype=np.float32)
        return vectors, multiclass_codes
    else:
        return vectors, code_list





# Tokenize and Pad notes text
def pad_notes(data, max_seq_length):
    data = pad_sequences(data, maxlen=max_seq_length, value=0)
    return data, data.shape[1]

def tokenize_text(text, MAX_NB_WORDS=None, OOV_token=None, word_frequency=2,
                  max_seq_length=100, verbose = True):
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=OOV_token)
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
    
    #tokenizer_json = tokenizer.to_json()
    
    padded_data = tf.keras.preprocessing.sequence.pad_sequences(data,
                                                         maxlen=max_seq_length,
                                                         padding='post')
    
    return padded_data, vocab, MAX_VOCAB, tokenizer





# Tokenize labels
def tokenize_labels(lang):
    # lang = list of sentences in a language
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang) 
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    max_label_length = tensor.shape[1]
    return tensor, lang_tokenizer, max_label_length


# In[11]:


ICD9_CODE_FIRST_LEVEL = ['001-139', '140-239', '240-279', '280-289', '290-319',
                         '320-389', '390-459', '460-519', '520-579', '580-629',
                         '630-679', '680-709','710-739', '740-759', '760-779',
                         '780-799', '800-999']


# ## Train Test Split

# In[12]:


X = df['TEXT']
y = df['ICD9_CODE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ## GRU(X) - GRU(Z) model (Level-1)

# ## Process Training data




X_train_p = process_text(X_train)





train_df = pd.DataFrame(data={'TEXT':X_train_p, 'Label':y_train}, index=None)





train_df.head()





train_df.shape





train_df_clean = train_df[train_df['TEXT']!=""].copy().reset_index().drop('index', axis=1)





train_df_clean.head()





X_train_final = np.asarray(train_df_clean['TEXT'].values)
labels_final = np.asarray(train_df_clean['Label'].values)


# ### Convert labels to Level-1 labels




train_labels = [replace_with_grandparent_codes(x, ICD9_CODE_FIRST_LEVEL) for x in labels_final]


# ## Tokenizing and padding text and labels to sequences




level = 1
BATCH_SIZE = 64
OOV_TOKEN = "<UNK>"
MAX_INPUT_LENGTH = 500
TEXT_WORD_FREQUENCY = 5


# In[22]:


input_seq, vocab, MAX_VOCAB, text_tokenizer = tokenize_text(text=X_train_final,
                                                            OOV_token=OOV_TOKEN,
                                                            word_frequency=TEXT_WORD_FREQUENCY,
                                                            max_seq_length=MAX_INPUT_LENGTH)





label_target, label_tokenizer, label_length = tokenize_labels(train_labels)





print(f'Shape of level_{level} label after tokenizing and padding is {label_target.shape}')
print(f'Maximum length of level_{level} tokenized labels is {label_length}')





train_dataset = tf.data.Dataset.from_tensor_slices((input_seq, label_target))
train_dataset = train_dataset.shuffle(buffer_size=input_seq.shape[0]).batch(BATCH_SIZE, drop_remainder=True)


# ## Checking batch of dataset




example_input_batch, example_target_batch = next(iter(train_dataset))
example_input_batch.shape, example_target_batch.shape


# ## Training for GRU(X) - GRU(Z) model




vocab_inp_size = len(text_tokenizer.word_index)+1
vocab_tar_size = len(label_tokenizer.word_index)+1
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

embedding_dim_inp = 200
enc_units = 50
embedding_dim_tar = 100
dec_units = 50
steps_per_epoch = input_seq.shape[0]//BATCH_SIZE


# ### GRU(X)




class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)

        gru_layer = tf.keras.layers.GRU(self.enc_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        
        
        self.bidirection_gru = tf.keras.layers.Bidirectional(gru_layer, merge_mode='ave',
                                                             input_shape=(self.batch_sz,
                                                                          self.embedding_dim))
        
    def call(self, x, hidden=None):
        x = self.embedding(x)
        output, h, c = self.bidirection_gru(x, initial_state = hidden)
        return output, (h+c)/2

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]





## Test Encoder Stack

encoder = Encoder(vocab_inp_size, embedding_dim_inp, enc_units, BATCH_SIZE)


# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_h = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder h vector shape: (batch size, units) {}'.format(sample_h.shape))


# ### GRU(Z)




class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.gru_layer = tf.keras.layers.GRU(self.dec_units,
                                             return_sequences=True,
                                             return_state=True,
                                             recurrent_initializer='glorot_uniform'
                                            )

        self.fc = tf.keras.layers.Dense(self.vocab_size)

    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        output, state = self.gru_layer(x, initial_state=initial_state)
        output = self.fc(output)
        return output, state





# Test decoder stack
decoder = Decoder(vocab_tar_size, embedding_dim_tar, dec_units, BATCH_SIZE)
sample_decoder_outputs, state = decoder(example_target_batch, initial_state=sample_h)
print("Decoder Outputs Shape: ", sample_decoder_outputs.shape)





optimizer = tf.keras.optimizers.Adam()

def loss_function(real, pred):
    # real shape = (BATCH_SIZE, max_length_output)
    # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    #print(f'real shape {real.shape}')
    #print(f'pred shape {pred.shape}')
    loss = cross_entropy(y_true=real, y_pred=pred)
    mask = tf.logical_not(tf.math.equal(real,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)  
    loss = mask* loss
    loss = tf.reduce_mean(loss)
    return loss





@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_state = encoder(inp, enc_hidden)

        dec_input = targ[ : , :-1 ] # Ignore <end> token
        real = targ[ : , 1: ]         # ignore <start> token

        logits, _ = decoder(dec_input, enc_state)
        
        loss = loss_function(real, logits)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss





def train_epochs(epochs=1, use_gpu=True):
    if use_gpu:
        device = '/GPU:0'
    else:
        device = '/CPU:0'
    with tf.device(device):
        for epoch in range(epochs):
            start = time.time()

            enc_hidden = encoder.initialize_hidden_state()
            total_loss = 0
            # print(enc_hidden[0].shape, enc_hidden[1].shape)

            for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
                batch_loss = train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))





start_time = time.time()
get_ipython().run_line_magic('memit', 'train_epochs(epochs=20)')
print("GPU time {} s".format(time.time()-start_time))


# ## Evaluation for GRU(X) - GRU(Z) model

# In[36]:


def evaluate_results(inputs, targets, inputs_tokenizer, target_tokenizer, max_input_length,
                     max_target_length, batch_size, target_codes, encoder_gru_units):
    
    X = process_text(inputs)
    
    test_df = pd.DataFrame(data={'TEXT':X, 'Label':targets}, index=None)
    test_df_clean = test_df[test_df['TEXT']!=""].copy().reset_index().drop('index', axis=1)
    
    X = np.asarray(test_df_clean['TEXT'].values)
    y = np.asarray(test_df_clean['Label'].values)
    
    
    y = [replace_with_grandparent_codes(label, target_codes) for label in y]

    X_test = inputs_tokenizer.texts_to_sequences(X)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test,
                                                           maxlen=max_input_length,
                                                           padding='post')
    
    y_true = target_tokenizer.texts_to_sequences(y)
    y_true = tf.keras.preprocessing.sequence.pad_sequences(y_true, maxlen=max_target_length,
                                                           padding='post')
    
    
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_true))
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    
    steps_per_epoch = X_test.shape[0]//batch_size
    inference_batch_size = batch_size
    result = []
    actual = []
    for (batch, (inp, targ)) in enumerate(test_dataset.take(steps_per_epoch)):
        y_pred = np.zeros((inference_batch_size, max_target_length), dtype=np.int16)
        enc_start_state = [tf.zeros((inference_batch_size, encoder_gru_units)),
                           tf.zeros((inference_batch_size, encoder_gru_units))]
        enc_output, dec_h = encoder(inp, enc_start_state)
        dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']] * inference_batch_size, 1)

        for t in range(max_target_length):
            #print(f'y_pred shape {y_pred.shape}')
            preds, dec_h = decoder(dec_input, dec_h)
            #print(f'preds shape {preds.shape}')
            preds = tf.squeeze(preds, axis=1)
            #print(f'preds shape {preds.shape}')
            predicted_id = tf.argmax(preds, axis=1).numpy()
            #print(f'predicted_id shape {predicted_id.shape}')
            y_pred[:, t] = predicted_id
            dec_input = tf.expand_dims(predicted_id, 1)
        result.append(y_pred)
        actual.append(targ)
    
    ground_truth = np.concatenate(actual, axis=0)
    prediction = np.concatenate(result, axis=0)
    return ground_truth, prediction


# In[37]:


y_true, y_pred = evaluate_results(inputs=X_test,
                                  targets=y_test, 
                                  inputs_tokenizer=text_tokenizer, 
                                  target_tokenizer=label_tokenizer, 
                                  max_input_length=MAX_INPUT_LENGTH,
                                  max_target_length=label_length,
                                  batch_size=BATCH_SIZE,
                                  target_codes=ICD9_CODE_FIRST_LEVEL,
                                  encoder_gru_units=enc_units)


# In[38]:


def evaluate_metrics(y_true, y_pred, label_tokenizer, verbose=1, title="Model"):
    label_codes = list(label_tokenizer.index_word.keys())
    y_pred_hot, _ = vectorize_icd_labels(labels=y_pred, code_list=label_codes)
    y_true_hot, _ = vectorize_icd_labels(labels=y_true, code_list=label_codes)
    f1 = f1_score(y_true_hot, y_pred_hot, average='micro')
    precision = precision_score(y_true_hot, y_pred_hot, average='micro')
    recall = recall_score(y_true_hot, y_pred_hot, average='micro')
    if verbose==1:
        if title is None:
            title = "Model"
        print("*******************************Summary for model metrics******************************")
        print("{:20s}\tF1-score\t{:2.3f}\tPrecision\t{:2.3f}\tRecall\t{:2.3f}".format(title, f1, precision, recall))
    return f1, precision, recall


# In[39]:


f1, precision , recall= evaluate_metrics(y_true, y_pred, label_tokenizer=label_tokenizer,
                                         title="GRU(X) - GRU(Z) (Level-{})".format(level))


# ## tfidf - atomic model  (Level-1)

# ### Preprocessing data

# In[40]:


# processing data for baseline model
X_train_b = process_text(X_train, add_start_end=False)


# In[41]:


train_df_b = pd.DataFrame(data={'TEXT':X_train_b, 'Label':y_train}, index=None)


# In[42]:


train_df_b_clean = train_df_b[train_df_b['TEXT']!=""].copy().reset_index().drop('index', axis=1)


# In[43]:


X_train_b = np.asarray(train_df_b_clean['TEXT'].values)
labels_final_b = np.asarray(train_df_b_clean['Label'].values)


# In[44]:


labels_b = [replace_with_grandparent_codes(x, ICD9_CODE_FIRST_LEVEL, add_start_end=False) for x in labels_final_b]


# In[45]:


max_number_features = 1000


# In[46]:


# TfidfVectorizer for text
vectorizer = TfidfVectorizer(max_features=max_number_features, stop_words='english', max_df=0.9 )  
X_train_vectors = vectorizer.fit_transform(X_train_b)


# In[47]:


# Vectorize labels
label_vectors, label_codes = vectorize_icd_labels(labels=labels_b,
                                                  code_list=ICD9_CODE_FIRST_LEVEL,
                                                  drop_single_class_codes=True)


# ### Training tfidf - atomic model

# In[48]:


model = LogisticRegression(solver='lbfgs', multi_class='ovr', class_weight='balanced', n_jobs=-1)

pipeline = Pipeline([
                ('tfidf', vectorizer),
                ('clf', MultiOutputRegressor(model))])


# In[49]:


start_time = time.time()
get_ipython().run_line_magic('memit', 'pipeline.fit(X_train_b, label_vectors)')
print("CPU time {} s".format(time.time()-start_time))


# ### Evaluating tfidf - atomic model

# In[50]:


def evaluate_metrics_tfidf(y_true, y_pred, verbose=1, title="Model"):
    f1 = f1_score(y_true, y_pred, average='micro')
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    if verbose==1:
        if title is None:
            title = "Model"
        print("*******************************Summary for model metrics******************************")
        print("{:20s}\tF1-score\t{:2.3f}\tPrecision\t{:2.3f}\tRecall\t{:2.3f}".format(title, f1, precision, recall))
    return f1, precision, recall


# In[51]:


X_test_b = process_text(X_test, add_start_end=False)


# In[52]:


test_df_b = pd.DataFrame(data={'TEXT':X_test_b, 'Label':y_test}, index=None)


# In[53]:


test_df_b_clean = test_df_b[test_df_b['TEXT']!=""].copy().reset_index().drop('index', axis=1)


# In[54]:


X_test_b_pre = np.asarray(test_df_b_clean['TEXT'].values)
y_test_b = np.asarray(test_df_b_clean['Label'].values)


# In[55]:


y_test_b_pre = [replace_with_grandparent_codes(x, label_codes, add_start_end=False) for x in y_test_b]


# In[56]:


y_test_true, _ = vectorize_icd_labels(labels=y_test_b_pre, code_list=label_codes)


# In[57]:


y_test_pred = pipeline.predict(X_test_b_pre)


# In[58]:


f1, precision, recall = evaluate_metrics_tfidf(y_test_true, y_test_pred, title="tfidf-atomic (Level-{})".format(level))

