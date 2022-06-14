import json

def update_data(n,pattern):
    with open('intents.json') as file:
        data = json.load(file)

    data['intents'][n]['patterns'].append(pattern)

    with open('intents.json','w') as file:
        json.dump(data,file)
    
    return "done"