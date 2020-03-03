from collections import Counter
import numpy as np
import tensorflow as tf
import pickle

#sys.path.append('D:/Documents/PythonCode/Text/')
#%%

def GetText(text, batch_size, seq_size, minOccurrence=0):
    text = text.split()

    word_counts = Counter(text)
    if minOccurrence > 0:
        word_counts = {k:v for k,v in word_counts.items() if v >= minOccurrence}
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)

    print('Vocabulary size', n_vocab)

    int_text = [vocab_to_int[w] for w in text if w in word_counts.keys()]
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
        
    def PredictIncrement(self, intWord, state, temp=1):
        intWord = tf.convert_to_tensor(
                [[intWord]], dtype=tf.float32)
        _, intPred, valState = self.call(intWord, state, temp=temp)
        intProbs = intPred.numpy()[0,0]
        return intProbs, valState

def get_word(int_pred, n_vocab, top=5):
    p = np.squeeze(int_pred)
    if top is not None:
        p[p.argsort()][:-top] = 0
    p = p / np.sum(p)
    word = np.random.choice(n_vocab, 1, p=p)[0]

    return word

#%%
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
    
def Predict(model, item, temp=1):
    predict(model, item['vocabToInt'], item['intToVocab'], item['nVocab'], temp=temp)


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
def SaveModel(outFile, model, intToVocab, vocabToInt, nVocab):
    model.save_weights(outFile)
    item = {'intToVocab':intToVocab, 'vocabToInt':vocabToInt, 'nVocab':nVocab, 
            'embeddingSize':model.embedding.output_dim, 'lstmSize':model.lstmSize}
    with open(outFile + 'Item', 'wb') as output:
        pickle.dump(item, output)
        
def LoadModel(inFile):
    inItem = inFile + 'Item'
    with open(inItem, 'rb') as input:
        item = pickle.load(input)
        
    model = RNN(nVocab=item['nVocab'], 
                embeddingSize=item['embeddingSize'], lstmSize=item['lstmSize'])
    model.load_weights(inFile)
    return model, item
