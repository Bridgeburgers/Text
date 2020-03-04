from collections import Counter
import numpy as np
import tensorflow as tf
import pickle

#sys.path.append('D:/Documents/PythonCode/Text/')
#%%

def GetText(text, batchSize, seqSize, minOccurrence=0):
    text = text.split()

    wordCounts = Counter(text)
    if minOccurrence > 0:
        wordCounts = {k:v for k,v in wordCounts.items() if v >= minOccurrence}
    sortedVocab = sorted(wordCounts, key=wordCounts.get, reverse=True)
    intToVocab = {k: w for k, w in enumerate(sortedVocab)}
    vocabToInt = {w: k for k, w in intToVocab.items()}
    nVocab = len(intToVocab)

    print('Vocabulary size', nVocab)

    intText = [vocabToInt[w] for w in text if w in wordCounts.keys()]
    numBatches = int(len(intText) / (seqSize * batchSize))
    inText = intText[:numBatches * batchSize * seqSize]
    outText = np.zeros_like(inText)
    outText[:-1] = inText[1:]
    outText[-1] = inText[0]
    inText = np.reshape(inText, (-1, seqSize))
    outText = np.reshape(outText, (-1, seqSize))
    return intToVocab, vocabToInt, nVocab, inText, outText


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
        output, stateH, stateC = self.lstm(embed, prevState)
        logits = self.dense(output)
        preds = tf.nn.softmax(temp * logits)
        return logits, preds, (stateH, stateC)

    def ZeroState(self, batchSize=1):
        return [tf.zeros([batchSize, self.lstmSize]),
                tf.zeros([batchSize, self.lstmSize])]
        
    def PredictIncrement(self, intWord, state, temp=1):
        intWord = tf.convert_to_tensor(
                [[intWord]], dtype=tf.float32)
        _, intPred, valState = self.call(intWord, state, temp=temp)
        intProbs = intPred.numpy()[0,0]
        return intProbs, valState

def GetWord(intPred, nVocab, top=5):
    p = np.squeeze(intPred)
    if top is not None:
        p[p.argsort()][:-top] = 0
    p = p / np.sum(p)
    word = np.random.choice(nVocab, 1, p=p)[0]

    return word

#%%
def predict(model, vocabToInt, intToVocab, nVocab, temp=1):

    valState = model.ZeroState(1)
    words = ['_START_']
    for word in words:
        intWord = tf.convert_to_tensor(
            [[vocabToInt[word]]], dtype=tf.float32)
        _, intPred, valState = model(intWord, valState, temp=temp)
    intPred = intPred.numpy()
    intWord = GetWord(intPred, nVocab)
    words.append(intToVocab[intWord])
    for _ in range(100):
        intWord = tf.convert_to_tensor(
            [[intWord]], dtype=tf.float32)
        _, intPred, valState = model(intWord, valState, temp=temp)
        intPred = intPred.numpy()
        intWord = GetWord(intPred, nVocab)
        words.append(intToVocab[intWord])
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
