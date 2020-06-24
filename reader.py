import tensorflow as tf

import os

words = []

# Read file
def read_data(filename):
    file_path = filename

    if os.path.isfile(file_path):
        with open(file_path) as f:
            raw = f.readlines()
    else:
        raise("[!] %s not found" % file_path)
    
    return raw

# Create word2idx and idx2word
# Sort words by frequency and index
def build_vocab(raw):
    
    for line in raw:
        words.extend(line.split())

    word2idx = {}
    word2idx['<eos>'] = 0

    #### random index
    # for word in words:
    #     if word not in word2idx:
    #         word2idx[word] = len(word2idx)
    # idx2word = dict(zip(word2idx.values(), word2idx.keys()))

    #### index by frequency
    counter = {}
    for word in words:
        if word not in counter:
            counter[word] = 0
        counter[word] += 1

    counter_sorted = sorted(counter.items(), key = lambda x:x[1], reverse = True)
    word2idx = {}
    word2idx['<eos>'] = 0

    i = 0
    for word, _ in counter_sorted:
        i = i+1
        word2idx[word] = i
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))

    return word2idx, idx2word

# Create dataset
def create_dataset(raw, word2idx, idx2word, batch_size):
    dataset = []
    seq_length = []

    # Word in data to idx
    for line in raw:
        data = []
        for word in line.split():
            data.append(word2idx[word])
        data.append(word2idx['<eos>'])
        dataset.append(data)

    dataX = []
    dataY = []
    max_seq_len = 0

    # Make dataset for 
    for data in dataset:
        x = data[:-1]
        y = data[1:] 
        dataX.append(x)
        dataY.append(y)
    
        cur_seq_len = len(x)
        seq_length.append(cur_seq_len)
        if cur_seq_len > max_seq_len:
            max_seq_len = cur_seq_len

    for data in dataX:
        while len(data)<max_seq_len:
            data.append(0)
        
    for data in dataY:
        while len(data)<max_seq_len:
            data.append(0)

    n_data = len(dataX)
    print(len(dataX))

    datasets = tf.data.Dataset.from_tensor_slices((dataX, dataY, seq_length)).batch(batch_size).repeat()
    iter = datasets.make_one_shot_iterator()
    datasetX, datasetY, seq_len_set = iter.get_next()

    return datasetX, datasetY, seq_len_set, max_seq_len, n_data
