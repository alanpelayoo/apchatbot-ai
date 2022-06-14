 
import json

from nltk.stem.lancaster import LancasterStemmer
from keras.models import Sequential 

#Importar capas
from keras.layers import Dense

#Importar regla de optimizacion
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD


stemmer = LancasterStemmer()

from data import DataProcessing #Import our class for data proccesing
from chat import chat


with open('intents.json') as file:
    data = json.load(file)
    
chatbot_data = DataProcessing(data) #Constructor
chatbot_data.start_process() # Generate data


model = Sequential()
model.add(Dense(128, input_shape=(len(chatbot_data.training[0]),), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(chatbot_data.output[0]), activation='softmax'))

opt = Adam(lr=0.05)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(chatbot_data.training, chatbot_data.output, epochs=1000, batch_size=5, verbose=1)

chat(chatbot_data,model,data)