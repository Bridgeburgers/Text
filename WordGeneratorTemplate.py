import os
from UrlToText import UrlToText, FilesToText
from WordGenerator import GetText, RNN, RNN2, GetWord, predict, Predict, train_func, SaveModel, LoadModel
import tensorflow as tf
import numpy as np
import timeit
import gc
#%%
#%%
seqSize = 200
batchSize = 32
embeddingSize = 1024
lstmSize = 2048
dropout = 0.3
gradientsNorm = 5 #norm to clip gradients
nEpochs = 100

#%%
#load COC text
path = 'E:/ArielS/TempData/TextData/CoC/'
files = os.listdir(path)
files = [path + f for f in files]
text = FilesToText(files, coc=True)
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
intToVocab, vocabToInt, nVocab, inText, outText = GetText(text, batchSize, seqSize, minOccurrence=6)
item = {'intToVocab': intToVocab, 'vocabToInt': vocabToInt, 'nVocab': nVocab,
        'embeddingSize': embeddingSize, 'lstmSize': lstmSize}
    
lenData = inText.shape[0]
stepsPerEpoch = lenData // batchSize
    
dataset = tf.data.Dataset.from_tensor_slices((inText, outText)).shuffle(100)
dataset = dataset.batch(batchSize, drop_remainder=True)

model = RNN(nVocab, embeddingSize, lstmSize, dropout)

state = model.ZeroState(batchSize)

optimizer = tf.keras.optimizers.Adam()
    
lossFunc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

file = 'D:/Documents/PythonCode/Models/MarkTwain3'
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
                predict(model, vocabToInt, intToVocab, nVocab, words=['_START_'])
                gc.collect()    
                
    if e % 5 == 0:
        SaveModel(file, model, intToVocab, vocabToInt, nVocab)
#%%
#SaveModel(file, model, intToVocab, vocabToInt, nVocab)
model, item = LoadModel(file, modelNum=1)
Predict(model, item)
vocabToInt, intToVocab, nVocab = item['vocabToInt'], item['intToVocab'], item['nVocab']

#%%
#use PredictIncrement
t = timeit.default_timer()
words = []
temp=1
intWord = vocabToInt['_START_']
word = '_START_'
state = model.ZeroState()
while word != '_END_':
    probs, state = model.PredictIncrement(intWord, state, temp)
    intWord = np.random.choice(range(nVocab), p=probs)
    word = intToVocab[intWord]
    words.append(word)
t = timeit.default_timer() - t
print(' '.join(words))
print(str(t) + ' seconds')

#%%
#%%
from PhraseSearch import Searcher
#%%
#create a searcher
c = 0.4
beta = 1000
probPower = 0.1
topWords = [300, 100]
inputPhrase = 'Well, it goes to show the importance of foresight and being prepared.'

searcher = Searcher(model, item, inputPhrase, c=c, beta=beta, probPower=probPower)
#%%
#test utility function with an original phrase from the text
originalPhrase = '_START_ well , it shows the value of looking ahead , and being ready for a thing when it comes . _END_'
searcher.PhraseUtilityProbs(originalPhrase)
#%%
#do a phrase search using MCTS
t = timeit.default_timer()
searcher.Search(nIterations=10000, resetTree=False, printWithEnd=True)
t = timeit.default_timer() - t
print(t)
d = searcher.phrases
