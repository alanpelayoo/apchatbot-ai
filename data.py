import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import json
stemmer = LancasterStemmer()

class DataProcessing:
    def __init__(self, data):
        self.data = data
        self.words = []
        self.labels = []
        self.sentence_x = []
        self.sentence_y = []

    def fill_lists(self):
        for intent in self.data['intents']:
            for pattern in intent['patterns']:
                wrds = nltk.word_tokenize(pattern)
                self.words.extend(wrds)
                self.sentence_x.append(wrds)
                self.sentence_y.append(intent["tag"])
                
            if intent['tag'] not in self.labels:
                self.labels.append(intent['tag'])

    def stem_list(self):
        self.fill_lists()
        self.words = [stemmer.stem(w.lower()) for w in self.words] #standarize words to base form.
        self.words = sorted(list(set(self.words)))  #Stemed vocabulary created.
        self.labels = sorted(self.labels)

        return self.words






