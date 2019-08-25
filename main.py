import nltk
import sys
import io
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import tflearn
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = [] #store the tag names
docs_x = [] #store the pattern data as a list
docs_y = [] #store the corresponding tag name for the pattern data
training = []
output = []
userData = {}  
normalChat = True

def populateLists(jsonData):
    global words
    global labels
    global docs_x
    global docs_y

    for intent in jsonData["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            #append ---- adds an element to a list
            #extend ---- concatenates the first list when another list
    
        if intent["tag"] not in labels:
            labels.append(intent["tag"])


def stemmingAList(inputList):
    # words = [stemmer.stem(w.lower()) for w in words if w != "?"]  same for do the following 
    newTemp = []

    for w in inputList:
        if w not in ["!","?","."] :
            newTemp.append(stemmer.stem(w.lower())) #removing question marks and perform stemming. Because of the inside list we have to iterate through the inside lists and do stemming after done the lowercase conversion
        
    return newTemp #replace the contents in words with the stemmed results


def sortingAList(inputList):
    newTemp = []

    newTemp = sorted(list(set(inputList)))

    return newTemp

def populateTrainingAndOutputLists():
    #training the bot
    #hard encoding
    global training
    global output

    output_empty = [0 for _ in range(len(labels))] #results --- [0,0,0]

    for x,doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = output_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open ("data.pickle","wb") as f:
       pickle.dump((words, labels, training, output),f)


def creatingTheModel():
    
    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])]) #input layer
    net = tflearn.fully_connected(net, 8) # adding a hidden layer to our network with 8 neurons
    net = tflearn.fully_connected(net, 8) 
    net = tflearn.fully_connected(net, len(output[0]),activation="softmax") #softmax give probability / output layer
    net = tflearn.regression(net) 

    modelTemp = tflearn.DNN(net) #DNN is a type of a network

    try:
        x
        modelTemp.load("model.tflearn")
        print("existing model found")
    except:
        print("model not found")
        modelTemp.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        modelTemp.save("model.tflearn")

        print("model created")

    return modelTemp

def bagOfWordsInUserInput(s, words):
    bag = [0 for _ in range(len(words))]

    sWords = nltk.word_tokenize(s)
    sWords = [stemmer.stem(word.lower()) for word in sWords]

    for entry in sWords:
        for i,wor in enumerate(words):
            if wor == entry:
                bag[i] = 1

    return numpy.array(bag)

def getAndStoreUserName():
    global userData
    print("userData not found")
    print("\n\tBot : Welcome ! I am nute !\n\tBot : I like to know your name first! What is your name ?")
    userName = input("\tYou : ")
    userData["name"] = userName

    with open ("userdata.pickle","wb") as f:
       pickle.dump(userData,f,protocol=pickle.HIGHEST_PROTOCOL)

def getResponseFromTheJson(tag):
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
            return responses

def confirmTheDetails(name,number):
    print('\tBot : Check the following and confirm. Please tell "confirm" if the details are right.\n\tTell "change" if you want to update the details. Tell "discard" if you want to terminate the saving.')

    print("\t\tContact s name: "+name+"\n\t\tNumber : "+number)

    res = input("\tYou : ")

    resBool = True

    if res == "confirm":
        resBool = True
    elif res == "change":
        print("\tBot : Change the name !")
        io.StringIO(name)
        name = input("\tYou : ")
        print("\tBot : Change the phone number !")
        io.StringIO(number)
        name = input("\tYou : ")
        confirmTheDetails(name,number)
    else:
        resBool = False
    
    return resBool


def addingANew():
    global normalChat
    normalChat = False

    print("\tBot : "+random.choice(getResponseFromTheJson("ss01")))
    contactName = input("\tYou : ")
    print("\tBot : "+random.choice(getResponseFromTheJson("ss02")))
    phoneNumber = input("\tYou : ")

    res = confirmTheDetails(contactName,phoneNumber)

    if res == True:
        print("\tBot : I memorized the new contact!"+res)
    else:
        print("\tBot : Discarded."+res)

    normalChat = True


def chat(model):

    print('\n\n\t\t************************* Type "quit" whenever you want to quit the chat ! *************************\n' )

    greetingString = random.choice(getResponseFromTheJson("greeting")).replace("$userName",userData["name"])
    
    print("\tBot : "+greetingString)

    while normalChat:
        
        userInput = input("\tYou : ")
        if userInput.lower() == "quit":
            break
        
        results = model.predict([bagOfWordsInUserInput(userInput, words)])[0]

        resultsIndex = numpy.argmax(results) #get the index of the highest probability

        tag = labels[resultsIndex]

        if tag == "add":
            addingANew()
        else:
            if results[resultsIndex] > 0.85:
                responses = getResponseFromTheJson(tag)

                if tag in ["greeting","userNameRequest"]:
                    print("\tBot : "+random.choice(responses).replace("$userName",userData["name"]))
                else:
                    print("\tBot : "+random.choice(responses))
           
            else:
                responses = getResponseFromTheJson("cannotUnderstand")
                print("\tBot : "+random.choice(responses))

        


def checkTheUserDataExistency():
    global userData
    
    try:
        with open ("userdata.pickle","rb") as f:
            userData = pickle.load(f)
        
    except:
        getAndStoreUserName()

    chat(model)


try:
    x
    with open ("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)
    print("data pickle found")
except:
    print("data pickle not found")
    populateLists(data)
    words = stemmingAList(words)
    words = sortingAList(words)
    labels = sorted(labels) #sorting the labels list
    populateTrainingAndOutputLists()



model = creatingTheModel()

checkTheUserDataExistency()
