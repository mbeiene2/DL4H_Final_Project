#!/usr/bin/env python
# coding: utf-8

# In[40]:


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


# In[35]:


seed = 1873
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# ## READING INPUT FILE

# In[2]:


filepath = "data/mimic3_data.pkl"


# In[3]:


with open(filepath, "rb") as new_file:
    df = pickle.load(new_file)


# In[4]:


df.head()


# ## Pre processing raw text and ICD 9 codes
def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=101):
    """Splits the input and labels into 3 sets"""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_size+test_size), random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(val_size+test_size), random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test
# In[5]:


def replace_with_grandparent_codes(string_codes, ICD9_CODE_FIRST_LEVEL):
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
    return '<start> ' + result + ' <end>' 

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
            result = result.strip()
            # adding a start and an end token to the sentence
            # so that the model know when to start and stop predicting.
            result = '<start> ' + result + ' <end>'
        processed_text.append(result)
    return processed_text


# In[36]:


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


# In[6]:


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


# In[7]:


# Tokenize labels
def tokenize_labels(lang):
    # lang = list of sentences in a language
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang) 
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    max_label_length = tensor.shape[1]
    return tensor, lang_tokenizer, max_label_length


# In[8]:


ICD9_CODE_FIRST_LEVEL = ['001-139','140-239','240-279','290-319', '320-389', '390-459','460-519', '520-579', '580-629',
                         '630-679', '680-709','710-739', '760-779', '780-789', '790-796', '797', '798', '799', '800-999']


# In[9]:


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

# In[10]:


X = df['TEXT']
y = df['ICD9_CODE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ## Process Training data

# In[11]:


X_train_p = process_text(X_train)


# In[12]:


level_1_labels = [replace_with_grandparent_codes(x, ICD9_CODE_FIRST_LEVEL) for x in y_train]


# In[13]:


level_2_labels = [replace_with_grandparent_codes(x, ICD9_CODE_SECOND_LEVEL) for x in y_train]


# In[14]:


train_df = pd.DataFrame(data={'TEXT':X_train_p, 'Level_1':level_1_labels, 'Level_2':level_2_labels}, index=None)


# In[15]:


train_df_clean = train_df[train_df['TEXT']!=""].copy().reset_index().drop('index', axis=1)


# In[16]:


X_train_final = np.asarray(train_df_clean['TEXT'].values)
level_1_final = np.asarray(train_df_clean['Level_1'].values)
level_2_final = np.asarray(train_df_clean['Level_2'].values)


# ## Tokenizing and padding text and labels to sequences

# In[17]:


BATCH_SIZE=64
OOV_TOKEN = "<UNK>"
MAX_INPUT_LENGTH = 500
TEXT_WORD_FREQUENCY = 5


# In[18]:


input_seq, vocab, MAX_VOCAB, text_tokenizer = tokenize_text(text=X_train_final,
                                                            OOV_token=OOV_TOKEN,
                                                            word_frequency=TEXT_WORD_FREQUENCY,
                                                            max_seq_length=MAX_INPUT_LENGTH)


# In[19]:


level_1_target, level_1_tokenizer, level_1_length = tokenize_labels(level_1_final)


# In[20]:


print(f'Shape of level_1 label after tokenizing and padding is {level_1_target.shape}')
print(f'Maximum length of level_1 tokenized labels is {level_1_length}')


# In[21]:


input_seq_train, input_seq_val, target_level1_train, target_level1_val = train_test_split(input_seq,
                                                                                          level_1_target,
                                                                                          test_size=0.2)


# In[22]:


train_level1_dataset = tf.data.Dataset.from_tensor_slices((input_seq_train, target_level1_train))
train_level1_dataset = train_level1_dataset.shuffle(buffer_size=input_seq_train.shape[0]).batch(BATCH_SIZE,
                                                                                                drop_remainder=True)


# In[23]:


val_level1_dataset = tf.data.Dataset.from_tensor_slices((input_seq_val, target_level1_val))
val_level1_dataset = val_level1_dataset.batch(BATCH_SIZE, drop_remainder=True)


# ## Checking batch of dataset

# In[24]:


example_input_batch, example_target_batch = next(iter(train_level1_dataset))
example_input_batch.shape, example_target_batch.shape


# ## Training for GRU(X) - GRU(Z) model

# In[25]:


vocab_inp_size = len(text_tokenizer.word_index)+1
vocab_tar_size = len(level_1_tokenizer.word_index)+1
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

embedding_dim_inp = 200
enc_units = 50
embedding_dim_tar = 100
dec_units = 50
steps_per_epoch = input_seq_train.shape[0]//BATCH_SIZE


# ### GRU(X)

# In[26]:


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


# In[27]:


## Test Encoder Stack

encoder = Encoder(vocab_inp_size, embedding_dim_inp, enc_units, BATCH_SIZE)


# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_h = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder h vector shape: (batch size, units) {}'.format(sample_h.shape))


# ### GRU(Z)

# In[28]:


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


# In[29]:


# Test decoder stack
decoder = Decoder(vocab_tar_size, embedding_dim_tar, dec_units, BATCH_SIZE)
sample_decoder_outputs, state = decoder(example_target_batch, initial_state=sample_h)
print("Decoder Outputs Shape: ", sample_decoder_outputs.shape)


# In[30]:


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


# In[31]:


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


# In[32]:


EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    # print(enc_hidden[0].shape, enc_hidden[1].shape)

    for (batch, (inp, targ)) in enumerate(train_level1_dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
  
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# ## Evaluation for GRU(X) - GRU(Z) model

# In[37]:


def evaluate_results(inputs, targets, inputs_tokenizer, target_tokenizer, max_input_length,
                     max_target_length, batch_size, target_codes, encoder_gru_units):
    
    X = process_text(inputs)
    y = [replace_with_grandparent_codes(label, target_codes) for label in targets]

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


# In[34]:


y_true, y_pred = evaluate_results(inputs=X_test,
                                  targets=y_test, 
                                  inputs_tokenizer=text_tokenizer, 
                                  target_tokenizer=level_1_tokenizer, 
                                  max_input_length=MAX_INPUT_LENGTH,
                                  max_target_length=level_1_length,
                                  batch_size=BATCH_SIZE,
                                  target_codes=ICD9_CODE_FIRST_LEVEL,
                                  encoder_gru_units=enc_units)


# In[41]:


def evaluate_metrics(y_true, y_pred, label_tokenizer, verbose=1, title="Model"):
    label_codes = list(label_tokenizer.index_word.keys())
    y_pred_hot = vectorize_icd_labels(labels=y_pred, code_list=label_codes)
    y_true_hot = vectorize_icd_labels(labels=y_true, code_list=label_codes)
    f1 = f1_score(y_true_hot, y_pred_hot, average='micro')
    precision = precision_score(y_true_hot, y_pred_hot, average='micro')
    recall = recall_score(y_true_hot, y_pred_hot, average='micro')
    if verbose==1:
        if title is None:
            title = "Model"
        print("*******************************Summary for model metrics******************************")
        print("{:20s}\tF1-score\t{:2.3f}\tPrecision\t{:2.3f}\tRecall\t{:2.3f}".format(title, f1, precision, recall))
    return f1, precision, recall


# In[42]:


f1, precision , recall= evaluate_metrics(y_true, y_pred, label_tokenizer=level_1_tokenizer,
                                         title="GRU(X) - GRU(Z)")

