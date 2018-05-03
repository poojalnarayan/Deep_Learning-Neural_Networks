import numpy as np
import keras
from keras.models import Sequential
from keras.layers import  Dropout, LSTM, Embedding, Bidirectional
from keras.layers import Dense
from sklearn.metrics import jaccard_similarity_score
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


class tweet_sentiment:

    def __init__(self):
        vocab = {}
        words_set = set()
        self.vocab = vocab
        self.words_set = words_set
        self.max_len = 50
        self.train_in = []
        self.train_out = []
        self.train_dev_in = []
        self.train_dev_out =[]
        self.test_in = []
        self.test_out = []
        self.predictions = []

    def read_data(self):
        with open("2018-E-c-En-train.txt", "r") as f:   # "test_input.txt"
            data = f.read()

        tweets = data.split("\n")
        tweets = tweets[1:]  #Remove the header
        f.close()
        for tweet in tweets:
            #print(tweet)
            fields = tweet.split("\t")
            #print(fields.__len__())
            tweet_text = fields[1]
            for t in tweet_text.split():
                self.words_set.add(t)
            #print(tweet_text)
            self.max_len = max(len(tweet_text.split()), self.max_len)

        print(self.words_set.__len__())
        self.vocab = {k: v for v, k in enumerate(self.words_set, start=2)}

        self.vocab['unknown_word'] = 1
        self.words_set.add('unknown_word')
        print(self.vocab.__len__())
        print(self.max_len)

        for tweet in tweets:
            #print(tweet)
            cur_tweet = np.zeros(self.max_len)
            fields = tweet.split("\t")
            #print(fields.__len__())
            tweet_text = fields[1]
            for i, t in enumerate(tweet_text.split()):
                cur_tweet[i] = self.vocab[t]
            self.train_in.append(cur_tweet)
            #print(tweet_text)
            cur_tweet_label = np.zeros(11)
            for i, l in enumerate(fields[2:]):
                cur_tweet_label[i] = l
            self.train_out.append(cur_tweet_label)

        #Todo- Take the max length of the words in the test data, train might be lesser!
        with open("2018-E-c-En-test.txt", "r") as f:   # Changed to test, Or Read dev data "2018-E-c-En-dev.txt"
            data = f.read()

        tweets = data.split("\n")
        tweets = tweets[1:]  #Remove the header
        f.close()
        for tweet in tweets:
            #print(tweet)
            cur_tweet = np.zeros(self.max_len)
            fields = tweet.split("\t")
            #print(fields.__len__())
            tweet_text = fields[1]
            for i, t in enumerate(tweet_text.split()):
                if t not in self.vocab:
                    cur_tweet[i] = self.vocab['unknown_word']
                else:
                    cur_tweet[i] = self.vocab[t]
            self.test_in.append(cur_tweet)
            #print(tweet_text)
            '''For dev remove this comment
            cur_tweet_label = np.zeros(11)
            for i, l in enumerate(fields[2:]):
                cur_tweet_label[i] = l
            self.test_out.append(cur_tweet_label)'''
        self.test_in = np.array(self.test_in)
        self.test_out = np.array(self.test_out)

    def build_model(self):
        model = Sequential()
        model.add(Embedding(len(self.words_set)+1, 128, input_length= self.max_len, embeddings_initializer='uniform'))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(11, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        kwargs = dict(batch_size=32)
        kwargs.update(x= np.array(self.train_in), y= np.array(self.train_out),
                      epochs=10)#, validation_data = (self.test_in, self.test_out ))

        model.fit(**kwargs)

        self.predictions = np.round(model.predict(np.array(self.test_in)))

        #Print jaccard similarity scores
        #js = jaccard_similarity_score(np.array(self.test_out), self.predictions)
        #print("js = ", js)


    def write_result(self):
        with open("2018-E-c-En-test.txt", "r") as f:   # Read dev data "2018-E-c-En-dev.txt"
            data = f.read()
        f.close()
        tweets=data.split("\n")
        header = tweets[0]
        tweets = tweets[1:]  # Remove the header

        with open("E-C_en_pred.txt", "w") as f1:   # Write to result file
            f1.write(header+ "\n")

            for i, tweet in enumerate(tweets):
                #print(tweet)
                cur_tweet = []
                fields = tweet.split("\t")

                cur_tweet.append(fields[0])
                cur_tweet.append("\t")
                cur_tweet.append(fields[1])

                for l in self.predictions[i]:
                    cur_tweet.append("\t")
                    cur_tweet.append( str(int(l)))

                cur_tweet.append("\n")
                f1.write(''.join(cur_tweet))
        f1.close()

ts = tweet_sentiment()
ts.read_data()
print("====================================================================================")
print(ts.train_in.__len__())
print(ts.train_out.__len__())

ts.build_model()
ts.write_result()