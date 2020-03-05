import urllib.request
import re
import numpy as np
#%%
def CleanCorpus(corpus):
    
    spaceStrings = ['\ufeff', '\r', '\n', ' ']
    punctuation = [',', ';', '-', ':', '_', '']
    
    corpus = corpus.lower()
    corpus = re.sub('[' + ''.join(spaceStrings) + ']+', ' ', corpus)
    for punc in punctuation:
        corpus = re.sub(punc, ' ' + punc, corpus)

    corpus = re.sub(r"([^ ])", r" \1", corpus)

    #separate dashes and underscores preceding characters
    corpus = re.sub(r'-(\w)', r'- \1', corpus)
    corpus = re.sub(r'_(\w)', r'_ \1', corpus)
    
    #separate parentheses
    corpus = re.sub(r'\((\w)', r'( \1', corpus)
    corpus = re.sub(r'(\w)\)', r'\1 )', corpus)
    
    #separate words contracted by apostrophes (and the apostrophe)
    corpus = re.sub(r"([^ ])' ", r"\1 ' ", corpus)
    corpus = re.sub(r"'([^ ]+) ", r" ' \1 ", corpus)

    #remove all double quotes
    corpus = re.sub('"', '', corpus)
    
    #convert all words containing numbers into _NUMBER_
    corpus = re.sub('[^ ]*[0-9]+[^ ]*', '_NUMBER_', corpus)
    
    

    corpus = re.sub('\. ', ' . _END_ _START_ ', corpus)
    corpus = re.sub('\? ', ' ? _END_ _START_ ', corpus)
    corpus = re.sub('! ', ' ! _END_ _START_ ', corpus)
    corpus = re.sub("\.' ", ".' _END_ _START_ ", corpus)
    #corpus = re.sub('\?" ', '?" _END_ _START_ ', corpus)
    #corpus = re.sub('!" ', '!" _END_ _START_ ', corpus)
        
    return corpus
    
#%%
def COCClean(text): 
    text = re.sub('@[0-9]+', '', text)
    text = re.sub('@', '', text)
    text = re.sub('<p>', '', text)
    text = re.sub('<h>', '', text)

    text = CleanCorpus(text)
#    text = re.sub('[' + ''.join(spaceStrings) + ']+', ' ', text)
#    
#    text = re.sub('\. ', ' . _END_ _START_ ', text)
#    text = re.sub('\? ', ' ? _END_ _START_ ', text)
#    text = re.sub('! ', ' ! _END_ _START_ ', text)
#    text = re.sub("\.' ", ".' _END_ _START_ ", text)
    
    return text
#%%
def UrlToText(urls):
    if not isinstance(urls, list) and not isinstance(urls, np.ndarray):
        urls = [urls]
    
    corpuses = []
    
    for url in urls:
        textStr = ''
        for line in urllib.request.urlopen(url):
            textStr = textStr + line.decode('utf-8')
        textStr = CleanCorpus(textStr)
        corpuses.append(textStr)
    
    text = ''
    for t in corpuses: text = text + t

    return text

#%%
def FilesToText(files, coc=True):
    if not isinstance(files, list) and not isinstance(files, np.ndarray):
        files = [files]
        
    corpuses = []
    
    for file in files:
        with open(file, 'r') as input:
            textStr = input.read()
        if coc:
            textStr = COCClean(textStr)
        else:
            textStr = CleanCorpus(textStr)
        corpuses.append(textStr)
        
    text = ''
    for t in corpuses:
        text = text + t
        
    return text
#%%
def CleanText(text):
    """
    Does roughly the inverse of CleanCorpus to make the output of LSTM readable
    """
    punctuation = [',', ';', '-', ':', '_', '']
    for punc in punctuation:
        text = re.sub(' ' + punc, punc, text)
        
    text = re.sub(" ", "", text)
    
    return text