"""
Since I loathe small talk, let's see if we can build a program that will do it for me.

"""

import nltk
import json
import pickle
from nltk.stem.lancaster import LancasterStemmer 

stemmer = LancasterStemmer()

# some helpers 
import numpy as np
import tflearn
import tensorflow as tf
import random

# import our chatbot's intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

# After loading JSON file, we can organize our documents, words, and classification classes.
words = []
classes = []
documents  = []
ignore_words = ['?']

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence 
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))

# stem and lower each word and remove duplicates
words = [steemmere.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

           # tokenize each word in the sentence
            w = nltk..word_tokenize(pattern)
            # add to our words list
            words.extend(w)
            # add to documents in our corpus
            documents.append((w, intent['tag']))

print(len(documents), " documents")
print(len(classes), " classes ", classes)
print(len(words), "unique stemmed words", words)


# Unfortunately, our data thus far will not work in Tensorflow. We need to transform it 
# further, from documents of words to tensors of numbers. 
training = []
output = []

# create an empty array for our output
output_empty = [0] * len(classes0)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []    # make sure to secure tha bag (SECURE THA BAG ALERT!!!)

   # list of tokenized words for the pattern 
    pattern_words = doc[0]

    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    # create a bag of words array 
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1] = 1)]

    training.append([bag, output_row])

# shuffle ouf features and turn into numpy array
random.shuffle(training)
training = np.array(training)

# create train and test sets
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# BUILDING OUR MODEL 

# reset underlying graph data

tf.reset_default_graph()
# build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression

# define model and setup tensorboard
model = tflearrn.DNN(net, tensorboard_dir='tflearn_logs')
# start training (apply gradient descent algorithms)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

# save ('pickle') our model and documents so the next notebook can use it
pickle.dump({'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open('training_data', 'wb'))

# After loading the same imports, we’ll un-pickle our model and documents as well as reload our intents file. 
# Remember our chatbot framework is separate from our model build — you don’t need to rebuild your model unless
# the intent patterns change. With several hundred intents and thousands of patterns the model could take several 
# minutes to buiild. 

# restore all of our data structures
data = pickle.load(open('training_data', 'rb'))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chatbot intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)

# load our saved model 
model.load('/model.tflearn')

"""
Before we can begin processing intents, we need a way to prooduce a bag of words
from user input. This is equivalentt to the steps for creating our training docs.

"""

def cleanup_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag existing in the sentence
def bow(sentence, words, show_datails=False):
    # tokenize the pattern 
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)

    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_datails:
                    print('found in bag: %s' % w)
    return np.array((bag)

# We're now ready to  build our response processor.
ERROR_THRESHOLD = 0.25

def classify(sentence):
    # generte probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter our predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by probability strength
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append(classes[r[0]], r[1]))

    # return tuple of intent and probability
    return return_list


def response(sentence, userId='123', show_details=False):
    results = classify(sentence)

    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    return print(random.choice(i['responses']))

            results.pop(0)

