import pandas
import numpy
import random
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import nltk

from write2json import update_data
def chat(chatbot_data,model,data):
    print("Chatbot")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        input_1 = pandas.DataFrame([bag_of_words(inp, chatbot_data.words)], dtype=float, index=['input'])
        results = model.predict([input_1])[0]

        results_index = numpy.argmax(results)
        tag = chatbot_data.labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))
        print(show_results(chatbot_data,results))

        answer = input("Is labelcorrect?")
        print(answer)
        if answer == "no":
            i=0
            for label_ in chatbot_data.labels:
                print(f"{i}: {label_}")
                i+=1

            answer2 = input("Choose the correct label (int)")
            print(update_data(int(answer2),inp))
            
def show_results(chatbot_data,results):
    error = 0.25
    results = [[i,r] for i,r in enumerate(results) if r>error]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((chatbot_data.labels[r[0]], str(r[1])))
    return return_list


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)