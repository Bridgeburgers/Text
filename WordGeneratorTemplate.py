from UrlToText import UrlToText
from WordGenerator import GetText, RNN, get_word, predict, Predict, train_func, SaveModel, LoadModel
import tensorflow as tf
import numpy as np
import timeit
#%%
#%%
seqSize = 300
batchSize = 16
embeddingSize = 256
lstmSize = 512
dropoutKeep = 0.7
gradientsNorm = 5 #norm to clip gradients
nEpochs = 50
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
SaveModel('E:/ArielS/Models/MarkTwain', model, intToVocab, vocabToInt, nVocab)
q, item = LoadModel('E:/ArielS/Models/MarkTwain')
Predict(q, item)
vocabToInt, intToVocab, nVocab = item['vocabToInt'], item['intToVocab'], item['nVocab']

#%%
#use PredictIncrement
t = timeit.default_timer()
words = []
temp=1
intWord = vocabToInt['_START_']
word = '_START_'
state = q.ZeroState()
while word != '_END_':
    probs, state = q.PredictIncrement(intWord, state, temp)
    intWord = np.random.choice(range(nVocab), p=probs)
    word = intToVocab[intWord]
    words.append(word)
t = timeit.default_timer() - t
print(' '.join(words))
print(str(t) + ' seconds')