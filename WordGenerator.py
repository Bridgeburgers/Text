import sys
from collections import Counter
import numpy as np
import tensorflow as tf

sys.path.append('D:/Documents/PythonCode/Text/')
from UrlToText import UrlToText

#%%
seqSize = 300
batchSize = 16
embeddingSize = 256
lstmSize = 256
dropoutKeep = 0.7
gradientsNorm = 5 #norm to clip gradients
nEpochs = 20
#%%

def GetText(text, batch_size, seq_size):
    text = text.split()

    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)

    print('Vocabulary size', n_vocab)

    int_text = [vocab_to_int[w] for w in text]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (-1, seq_size))
    out_text = np.reshape(out_text, (-1, seq_size))
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text


class RNN(tf.keras.Model):
    def __init__(self, nVocab=None, embeddingSize=None, lstmSize=None):
        super(RNN, self).__init__()
        if nVocab is not None and embeddingSize is not None and lstmSize is not None:
            self.lstmSize = lstmSize
            self.embedding = tf.keras.layers.Embedding(
                nVocab, embeddingSize)
            self.lstm = tf.keras.layers.LSTM(
                lstmSize, return_state=True, return_sequences=True)
            self.dense = tf.keras.layers.Dense(nVocab)

    def call(self, x, prevState, temp=1):
        embed = self.embedding(x)
        output, state_h, state_c = self.lstm(embed, prevState)
        logits = self.dense(output)
        preds = tf.nn.softmax(temp * logits)
        return logits, preds, (state_h, state_c)

    def ZeroState(self, batchSize=1):
        return [tf.zeros([batchSize, self.lstmSize]),
                tf.zeros([batchSize, self.lstmSize])]

def get_word(int_pred, n_vocab, top=5):
    p = np.squeeze(int_pred)
    if top is not None:
        p[p.argsort()][:-top] = 0
    p = p / np.sum(p)
    word = np.random.choice(n_vocab, 1, p=p)[0]

    return word

def predict(model, vocab_to_int, int_to_vocab, n_vocab, temp=1):

    val_state = model.ZeroState(1)
    words = ['_START_']
    for word in words:
        int_word = tf.convert_to_tensor(
            [[vocab_to_int[word]]], dtype=tf.float32)
        _, int_pred, val_state = model(int_word, val_state, temp=temp)
    int_pred = int_pred.numpy()
    int_word = get_word(int_pred, n_vocab)
    words.append(int_to_vocab[int_word])
    for _ in range(100):
        int_word = tf.convert_to_tensor(
            [[int_word]], dtype=tf.float32)
        _, int_pred, val_state = model(int_word, val_state, temp=temp)
        int_pred = int_pred.numpy()
        int_word = get_word(int_pred, n_vocab)
        words.append(int_to_vocab[int_word])
    print(' '.join(words))


@tf.function
def train_func(inputs, targets, model, state, loss_func, optimizer):
  with tf.GradientTape() as tape:
      logits, _, state = model(inputs, state)

      loss = loss_func(targets, logits)

      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(
          zip(gradients, model.trainable_variables))
      return loss

#%%
#load Mark Twain text  
urls = [
        'https://digihum.mcgill.ca/wp-content/uploads/2014/02/Connecticut_Yankee.txt',
        'https://digihum.mcgill.ca/wp-content/uploads/2014/02/Huckleberry_Finn.txt',
        'https://digihum.mcgill.ca/wp-content/uploads/2014/02/Tom_Sawyer.txt',
        'https://digihum.mcgill.ca/wp-content/uploads/2014/02/Mississippi.txt',
        'https://digihum.mcgill.ca/wp-content/uploads/2014/02/Innocents_Abroad.txt'
        ]

text = UrlToText(urls) 
#%%
intToVocab, vocabToInt, nVocab, inText, outText = GetText(text, batchSize, seqSize)
    
lenData = inText.shape[0]
stepsPerEpoch = lenData // batchSize
    
dataset = tf.data.Dataset.from_tensor_slices((inText, outText)).shuffle(100)
dataset = dataset.batch(batchSize, drop_remainder=True)

model = RNN(nVocab, embeddingSize, lstmSize)

state = model.ZeroState(batchSize)

optimizer = tf.keras.optimizers.Adam()
    
lossFunc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#%%
for e in range(nEpochs):
    state = model.ZeroState(batchSize)

    for (batch, (inputs, targets)) in enumerate(dataset.take(stepsPerEpoch)):
        loss = train_func(inputs, targets, model, state, lossFunc, optimizer)

        if batch % 50 == 0:
            print('Epoch: {}/{}'.format(e, nEpochs),
                  'Batch: {}'.format(batch),
                  'Loss: {}'.format(loss.numpy()))
                    
            if batch % 150 == 0:
                predict(model, vocabToInt, intToVocab, nVocab)

#%%
                
