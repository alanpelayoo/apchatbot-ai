import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import random
import json

with open('intents.json') as file:
    data = json.load(file)

stemmer = LancasterStemmer()
words = []
labels = []
sentence_x = []
sentence_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        sentence_x.append(wrds)
        sentence_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words] #standarize words to base form.
words = sorted(list(set(words)))  #Stemed vocabulary created.

labels = sorted(labels)

training = []
output = []

out_empty = [ 0 for _ in range(len(labels))]


for x,doc in enumerate(sentence_x):
    bag = []
    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(sentence_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)
print(training)
