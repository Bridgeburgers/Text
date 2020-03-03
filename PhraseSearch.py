import numpy as np
import pandas as pd
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
    """
    params
    ------
    
    phrase: str
        the phrase whose embedding is to be compared with the input phrase   
    inputVector: np.ndarray
        the BERT embedded vector of the input phrase
    probs: list
        the sequence of probability transitions required for input phrase
    beta: float
        the weighting given to content similarity
    """
    if len(probs)==0:
        return 0
    phraseIntegrity = np.power(np.prod(probs), 1/len(probs))
    
    phraseEmbedding = np.array( transformer.encode([phrase]) )[0]
    contentSimilarity = CosSimilarity(phraseEmbedding, inputVector)
    
    #get weighted harmonic mean, weighting cosSimilarity with beta
    utility = (1 + beta) / (1/phraseIntegrity + beta/contentSimilarity)
    
    return utility

#%%
class Node:
    
    def __init__(self, word, prob, state, phraseProbs, phrase=[]):
        self.word = word
        self.phrase = phrase + [word]
        self.p = prob #probability of children nodes
        self.childNodes = {}
        self.Q = 0
        self.N = 0 #number of visits
        self.childVisits = np.zeros(len(prob))
        self.childQs = np.zeros(len(prob))
        self.state = state
        self.root = False
        self.phraseProbs = phraseProbs
        
#%%  
class Searcher:
    
    def __init__(self, model, item, inputPhrase, c=0.5, beta=1, probPower=1):
        
        self.vocabToInt = item['vocabToInt']
        self.intToVocab = item['intToVocab']
        self.nVocab = item['nVocab']
        self.model = model
        self.beta = beta
        self.c = c
        self.probPower = probPower
        self.inputPhrase = inputPhrase
        
        self.startToken = self.vocabToInt['_START_']
        self.endToken = self.vocabToInt['_END_']
        
        startingProb, startingState = model.PredictIncrement(
                self.startToken, model.ZeroState())
        
        self.tree = Node(self.startToken, startingProb, startingState, [], phrase=[])
        self.tree.root = True
        
        #create the input embedding vector by embedding inputPhrase
        self.inputVector = transformer.encode([inputPhrase])[0]
        
        self.phrases = pd.DataFrame({'phrase':[], 'utility':[]})
    
    def PhraseUtility(self, textPhrase, phraseProbs):
        if '_START_' in textPhrase:
            textPhrase.remove('_START_')
        if '_END_' in textPhrase:
            textPhrase.remove('_END_')
        textPhrase = ' '.join(textPhrase)
        utility = Utility(textPhrase, self.inputVector, phraseProbs, self.beta)
        return utility
    
    def ChildUtilities(self, node):
        childUtilities = node.childQs + self.c * np.power(node.p, self.probPower) *\
            np.sqrt(np.sum(node.childVisits)) / (1 + node.childVisits)
            
        return childUtilities
    
    def SearchNode(self, node):
        node.N += 1
        
        #if currentNode is a leaf, evaluate and backpropagate
        if node.N == 1:
            #evaluate utility
            textPhrase = [self.intToVocab[j] for j in node.phrase]
            phraseUtility = self.PhraseUtility(textPhrase.copy(), node.phraseProbs)
            
            self.phrases = self.phrases.append(
                    {'phrase': ' '.join(textPhrase), 'utility': phraseUtility},
                    ignore_index=True)
            
            if node.phrase[-1] == self.endToken:
                #make sure this node doesn't get revisited by setting node Q very low
                node.Q = -9999
            else:
                node.Q = phraseUtility
                
            return node.Q
                    
        #keep traversing
        #get child with highest childUtility
        childUtilities = self.ChildUtilities(node)
        nextWord = np.argmax(childUtilities)
        nextProb = node.p[nextWord]
        childPhraseProbs = node.phraseProbs + [nextProb]
        if node.childVisits[nextWord] == 0:
            #create state and prob objects by incrementing the generator model
            childProbs, childState = self.model.PredictIncrement(nextWord, node.state)
            childNode = Node(nextWord, childProbs, childState, childPhraseProbs, node.phrase)
            node.childNodes[nextWord] = childNode
            
        node.childVisits[nextWord] += 1
        childQ = self.SearchNode(node.childNodes[nextWord])
        node.childQs[nextWord] = node.childNodes[nextWord].Q
        return childQ
        
    def Search(self, nIterations=1000, resetTree=False, printWithEnd=True):
        if resetTree:
            self.phrases = pd.DataFrame({'phrase':[], 'utility':[]})
            startingProb, startingState = self.model.PredictIncrement(
                self.startToken, self.model.ZeroState())
        
            self.tree = Node(self.startToken, startingProb, startingState, [], phrase=[])
            self.tree.root = True
            
        for _ in range(nIterations):
            self.SearchNode(self.tree)
            
        self.BestPhrase(withEnd=printWithEnd)
            
    def PhraseWithEnd(self):
        return self.phrases[self.phrases.phrase.str.contains('_END_')]
        
    def BestPhrase(self, withEnd=False):
        if withEnd:
            d = self.PhraseWithEnd()
        else:
            d = self.phrases
        ind = d.utility.idxmax()
        print(d.phrase.loc[ind])
        print(str(d.utility.loc[ind]))