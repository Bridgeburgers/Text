import urllib.request
import re
import numpy as np

#%%
def CleanCorpus(corpus):
    
    spaceStrings = ['\ufeff', '\r', '\n', ' ']
    punctuation = [',', ';', '-', ':', '_']
    
    corpus = corpus.lower()
    corpus = re.sub('[' + ''.join(spaceStrings) + ']+', ' ', corpus)
    for punc in punctuation:
        corpus = re.sub(punc, ' ' + punc, corpus)

    #separate dashes and underscores preceding characters
    corpus = re.sub(r'-(\w)', r'- \1', corpus)
    corpus = re.sub(r'_(\w)', r'_ \1', corpus)
    
    #separate parentheses
    corpus = re.sub(r'\((\w)', r'( \1', corpus)
    corpus = re.sub(r'(\w)\)', r'\1 )', corpus)
    

    corpus = re.sub('"', '', corpus)

    corpus = re.sub('\. ', ' . _END_ _START_ ', corpus)
    corpus = re.sub('\? ', ' ? _END_ _START_ ', corpus)
    corpus = re.sub('! ', ' ! _END_ _START_ ', corpus)
    #corpus = re.sub('\." ', '." _END_ _START_ ', corpus)
    #corpus = re.sub('\?" ', '?" _END_ _START_ ', corpus)
    #corpus = re.sub('!" ', '!" _END_ _START_ ', corpus)
        
    return corpus

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