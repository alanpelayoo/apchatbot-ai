
import numpy
import tensorflow 
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pandas

from keras.models import Sequential 

#Importar capas
from keras.layers import Dense

#Importar regla de optimizacion
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD


stemmer = LancasterStemmer()

from data import DataProcessing #Import our class for data proccesing
from write2json import update_data


with open('intents.json') as file:
    data = json.load(file)

chatbot_data = DataProcessing(data) #Constructor
chatbot_data.start_process() # Generate data


model = Sequential()
model.add(Dense(80, input_shape=(len(chatbot_data.training[0]),), activation='relu'))

model.add(Dense(80, activation='relu'))

model.add(Dense(len(chatbot_data.output[0]), activation='softmax'))

opt = Adam(lr=0.05)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(chatbot_data.training, chatbot_data.output, epochs=1000, batch_size=5, verbose=1)


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        input_1 = pandas.DataFrame([bag_of_words(inp, chatbot_data.words)], dtype=float, index=['input'])
        results = model.predict([input_1])[0]

        results_index = numpy.argmax(results)
        print(results_index)
        tag = chatbot_data.labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))
        print(classify_local(inp))

        answer = input("Is labelcorrect?")
        print(answer)
        if answer == "no":
            i=0
            for label_ in chatbot_data.labels:
                print(f"{i}: {label_}")
                i+=1

            answer2 = input("Choose the correct label (int)")
            print(update_data(int(answer2),inp))
            
def classify_local(sentence):
    ERROR_THRESHOLD = 0.25
    
    # generate probabilities from the model
    input_data = pandas.DataFrame([bag_of_words(sentence, chatbot_data.words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((chatbot_data.labels[r[0]], str(r[1])))
    # return tuple of intent and probability
    return return_list


chat()