
import numpy
import tflearn
import tensorflow
import random
import json

from data import DataProcessing

with open('intents.json') as file:
    data = json.load(file)

objt_1 = DataProcessing(data) #Constructor
objt_1.start_process() # Generate data

print(objt_1.training[0])
print(objt_1.output)

