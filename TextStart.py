import urllib.request
import re
import timeit
from collections import Counter
import numpy as np
import os

import tensorflow as tf
#%%
tf.reset_default_graph()

#%%
spaceStrings = ['\ufeff', '\r', '\n', ' ']
punctuation = [',', ';', '-', ':']

def CleanCorpus(corpus):
    corpus = corpus.lower()
    corpus = re.sub('[' + ''.join(spaceStrings) + ']+', ' ', corpus)
    for punc in punctuation:
        corpus = re.sub(punc, ' ' + punc, corpus)

    corpus = re.sub('"', '', corpus)

    corpus = re.sub('\. ', ' . _END_ _START_ ', corpus)
    corpus = re.sub('\? ', ' ? _END_ _START_ ', corpus)
    corpus = re.sub('! ', ' ! _END_ _START_ ', corpus)
    #corpus = re.sub('\." ', '." _END_ _START_ ', corpus)
    #corpus = re.sub('\?" ', '?" _END_ _START_ ', corpus)
    #corpus = re.sub('!" ', '!" _END_ _START_ ', corpus)
        
    return corpus
#%%

urls = [
        'https://digihum.mcgill.ca/wp-content/uploads/2014/02/Connecticut_Yankee.txt',
        'https://digihum.mcgill.ca/wp-content/uploads/2014/02/Huckleberry_Finn.txt',
        'https://digihum.mcgill.ca/wp-content/uploads/2014/02/Tom_Sawyer.txt',
        'https://digihum.mcgill.ca/wp-content/uploads/2014/02/Mississippi.txt',
        'https://digihum.mcgill.ca/wp-content/uploads/2014/02/Innocents_Abroad.txt'
        ]

#%%
t = timeit.default_timer()
corpuses = []

for url in urls:
    textStr = ''
    for line in urllib.request.urlopen(url):
        textStr = textStr + line.decode('utf-8')
    textStr = CleanCorpus(textStr)
    corpuses.append(textStr)
    
elapsed = timeit.default_timer() - t


text = ''
for t in corpuses: text = text + t

#%%
def GetDataFromText(text, batchSize, seqSize):

    text = text.split()
    wordCounts = Counter(text)
    sortedVocab = sorted(wordCounts, key=wordCounts.get, reverse=True)
    intToVocab = {k:w for k,w in enumerate(sortedVocab)}
    vocabToInt = {w:k for k,w in intToVocab.items()}
    nVocab = len(intToVocab)
    
    startKey = vocabToInt['_START_']
    endKey = vocabToInt['_END_']
    
    intText = [vocabToInt[w] for w in text]
    nBatches = int(len(intText) / (seqLength * batchSize))
    inText = intText[:nBatches * batchSize * seqLength]
    
    #set outText to be the next word in inText
    outText = np.zeros_like(inText)
    outText[:-1] = inText[1:]
    outText[-1] = inText[0]
    
    inText = np.reshape(inText, (batchSize, -1))
    outText = np.reshape(outText, (batchSize, -1))
    
    return intToVocab, vocabToInt, nVocab, inText, outText

def GetBatches(inText, outText, batchSize, seqSize):
    numBatches = np.prod(inText.shape) // (seqSize * batchSize)
    for i in range(0, numBatches * seqSize, seqSize):
        yield inText[:, i:i+seqSize], outText[:, i:i+seqSize]

#%%
def Model(batch_size, seq_size, embedding_size, lstm_size, keep_prob, n_vocab, reuse=False):
    with tf.variable_scope('LSTM', reuse=reuse):
        in_op = tf.placeholder(tf.int32, [None, seq_size])
        out_op = tf.placeholder(tf.int32, [None, seq_size])
        embedding = tf.get_variable('embedding_weights', [n_vocab, embedding_size])
        embed = tf.nn.embedding_lookup(embedding, in_op)
        lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
        initial_state = lstm.zero_state(batch_size, dtype=tf.float32)
    
    output, state = tf.nn.dynamic_rnn(lstm, embed, initial_state=initial_state, dtype=tf.float32)
    logits = tf.layers.dense(output, n_vocab, reuse=reuse)
    preds = tf.nn.softmax(logits)
    
    return in_op, out_op, lstm, initial_state, state, preds, logits


#%%
def GetLossAndTrainOp(outOp, logits, gradientsNorm):
    lossOp = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=outOp, logits=logits))
    
    trainableVars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(lossOp, trainableVars), gradientsNorm)
    opt = tf.train.AdamOptimizer()
    trainOp = opt.apply_gradients(zip(grads, trainableVars))
    
    return lossOp, trainOp

#%%
batchSize = 64
bufferSize = 10000
embeddingDim = 256
epochs = 10
seqLength = 300
gradientsNorm = 5
examplesPerEpoch = len(text)//seqLength
#lr = 0.001 #will use default for Adam optimizer
rnnUnits = 760
#vocabSize = nVocab
dropoutKeep = 0.8

checkpointPath = 'D:/Documents/TempWts'

#%%
intToVocab, vocabToInt, nVocab, inText, outText = GetDataFromText(text, batchSize, seqLength)

#for training
inOp, outOp, lstm, initialState, state, preds, logits = Model(
        batchSize, seqLength, embeddingDim, rnnUnits, dropoutKeep, nVocab)

#for inference
valInOp, _, _, valInitialState, valState, valPreds, _ = Model(
        1, 1, embeddingDim, rnnUnits, dropoutKeep, nVocab, reuse=True)

lossOp, trainOp = GetLossAndTrainOp(outOp, logits, gradientsNorm)

#%%
sess = tf.Session()
saver = tf.train.Saver()

if not os.path.exists(checkpointPath):
    os.mkdir(checkpointPath)
    
sess.run(tf.global_variables_initializer())

#%%

iteration = 0

for e in range(epochs):
    batches = GetBatches(inText, outText, batchSize, seqLength)
    newState = sess.run(initialState)

    for x, y in batches:
          iteration += 1
          loss, new_state, _ = sess.run(
              [lossOp, state, trainOp],
              feed_dict={inOp: x, outOp: y, initialState: newState})
          if iteration % 100 == 0:
              print('Epoch: {}/{}'.format(e, epochs),
                    'Iteration: {}'.format(iteration),
                    'Loss: {:.4f}'.format(loss))
#          if iteration % 1000 == 0:
#              predict(FLAGS.initial_words, FLAGS.predict_top_k,
#                      sess, val_in_op, val_initial_state,
#                      val_preds, val_state, n_vocab,
#                      vocab_to_int, int_to_vocab)
              saver.save(
                  sess,
                  os.path.join(checkpointPath, 'model-{}.ckpt'.format(iteration)))
#%%
def predict(initial_words, predict_top_k, sess, in_op,
            initial_state, preds, state, n_vocab, vocab_to_int, int_to_vocab):
    new_state = sess.run(initial_state)
    words = initial_words
    samples = [w for w in words]

    for word in words:
        x = np.zeros((1, 1))
        x[0, 0] = vocab_to_int[word]
        pred, new_state = sess.run([preds, state], feed_dict={in_op: x, initial_state: new_state})           
        
#%%
model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabSize, embeddingDim, batch_input_shape = [batchSize, None]),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(rnnUnits, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(dropout),
        #tf.keras.layers.LSTM(rnnUnits, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        #tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(vocabSize, activation='softmax')
        ])

#%%
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

history = model.fit(inText, outText, epochs=epochs, verbose=2)
