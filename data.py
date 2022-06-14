import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
stemmer = LancasterStemmer()

class DataProcessing:
    def __init__(self, data):
        self.data = data
        self.words = []
        self.labels = []
        self.sentence_x = []
        self.sentence_y = []
        self.training = []
        self.output = []

    def fill_lists(self):
        print("step 1")
        print(self.words,self.training)
        for intent in self.data['intents']:
            for pattern in intent['patterns']:
                wrds = nltk.word_tokenize(pattern)
                self.words.extend(wrds)
                self.sentence_x.append(wrds)
                self.sentence_y.append(intent["tag"])
                
            if intent['tag'] not in self.labels:
                self.labels.append(intent['tag'])
        self.stem_list()

    def stem_list(self):
        print("step 2")
        self.words = [stemmer.stem(w.lower()) for w in self.words] #standarize words to base form.
        self.words = sorted(list(set(self.words)))  #Stemed vocabulary created.
        self.labels = sorted(self.labels)
        self.bag_of_words()

    

    def bag_of_words(self):
        print("step 3")
        out_empty = [ 0 for _ in range(len(self.labels))]

        for x,doc in enumerate(self.sentence_x):
            bag = []
            wrds = [stemmer.stem(w.lower()) for w in doc]

            for w in self.words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[self.labels.index(self.sentence_y[x])] = 1

            self.training.append(bag)
            self.output.append(output_row)
  
        self.training = numpy.array(self.training)
        self.output = numpy.array(self.output)

    def start_process(self):
        self.fill_lists()
       
        
        







