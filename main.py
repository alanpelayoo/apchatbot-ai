
import numpy
import tflearn
import tensorflow as tf
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

from data import DataProcessing #Import our class for data proccesing

with open('intents.json') as file:
    data = json.load(file)

chatbot_data = DataProcessing(data) #Constructor
chatbot_data.start_process() # Generate data



tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(chatbot_data.training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(chatbot_data.output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

model.fit(chatbot_data.training, chatbot_data.output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


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

        results = model.predict([bag_of_words(inp, chatbot_data.words)])
        results_index = numpy.argmax(results)
        tag = chatbot_data.labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()