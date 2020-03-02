import numpy as np
#%%
#load BERT embedding model
from sentence_transformers import SentenceTransformer
transformer = SentenceTransformer('bert-base-nli-mean-tokens')
#%%
def CosSimilarity(x,y):
    dotProduct = sum(x * y)
    xMag = np.sqrt(sum(pow(x, 2)))
    yMag = np.sqrt(sum(pow(y,2)))
    cos = dotProduct / xMag / yMag
    return cos

def Utility(phrase, inputVector, probs, beta=1):
    phraseIntegrity = np.power(np.prod(probs), 1/np.len(probs))
    
    phraseEmbedding = np.array( transformer.encode([phrase]) )[0]
    contentSimilarity = CosSimilarity(phraseEmbedding, inputVector)
    
    #get weighted harmonic mean, weighting cosSimilarity with beta
    utility = (1 + beta) / (1/phraseIntegrity + beta/contentSimilarity)
    
    return utility