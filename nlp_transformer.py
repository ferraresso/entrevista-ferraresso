from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from tqdm import tqdm
import pandas as pd
import numpy as np

def nlp_clean_characters(data):
    new_str = data.lower()
    for c in ';"!?.,()=#$/][-º\'':
        new_str.replace(c, ' ')
    new_str = new_str.replace('á', 'a')
    new_str = new_str.replace('é', 'e')
    new_str = new_str.replace('í', 'i')
    new_str = new_str.replace('ó', 'o')
    new_str = new_str.replace('ú', 'u')

    for c in '0123456789':
        new_str.replace(c, '#')

    return new_str


class NLPCleaner():
    def __init__(self, stopwords=[], with_lemma=True):
        self.custom_stopword = stopwords
        self.with_lemma = with_lemma
        if self.with_lemma:
            lemma = pd.read_csv('lemmatization-es.csv')
            self.lemma_dic = {}
            lemma = lemma[['word','lemma']].values
            for i in tqdm(range(len(lemma))):
                self.lemma_dic[lemma[i,0]] = lemma[i,1]
        
    def get_lemma(self, word):
        if self.lemma_dic.get(word) is None:
            return word
        else:
            return self.lemma_dic[word]
            
    def fit(self, X, y=None):
        return self
    
    def transform(self, data, y=None):
        return self.nlp_clean(data, max_len=2000)
    
    def nlp_clean(self, data, max_len=200):
        reg_tokenizer = RegexpTokenizer(r'[A-Za-z]\w+')
        stopword_set = set(stopwords.words('spanish'))
        
        for c in self.custom_stopword:
            stopword_set.add(c)
            
        max_words = 0
        new_data = []
        for i,d in enumerate(data):
            try:
                new_str = d
                new_str = nlp_clean_characters(new_str)
                dlist = reg_tokenizer.tokenize(new_str)
                dlist = [d for d in dlist if d not in stopword_set]
                dlist = [d for d in dlist if len(d) > 1]
                if self.with_lemma:
                    dlist = [self.get_lemma(d) for d in dlist]
                new_data.append(' '.join(dlist[:max_len]))
            except AttributeError:
                new_data.append('sintexto')
        return new_data
    
    
class StringFeatures():
    def fit(self, X, y=None):
        return self
    
    def transform(self, data, y=None):
        res = []
        for d in data:
            res.append([
                        len(d.split()),
                        len(set(d.split())),
                        len(set(d.split()))/len(d.split())
                        ])
                
        return np.asarray(res)