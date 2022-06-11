
import numpy
import tflearn
import tensorflow
import random
import json

from data import DataProcessing

with open('intents.json') as file:
    data = json.load(file)

objt_1 = DataProcessing(data)
objt_1.stem_list()

print(objt_1.words)
# training = []
# output = []

# out_empty = [ 0 for _ in range(len(labels))]


# for x,doc in enumerate(sentence_x):
#     bag = []
#     wrds = [stemmer.stem(w.lower()) for w in doc]

#     for w in words:
#         if w in wrds:
#             bag.append(1)
#         else:
#             bag.append(0)

#     output_row = out_empty[:]
#     output_row[labels.index(sentence_y[x])] = 1

#     training.append(bag)
#     output.append(output_row)

# training = numpy.array(training)
# output = numpy.array(output)
# print(training)
