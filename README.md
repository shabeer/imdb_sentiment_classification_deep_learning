
# Sentiment Classification using Deep Learning

## Aim : To classify sentiment of movie reviews in imdb dataset as positive or negative. 

### Results : Please check at the end of notebook

#### Tools used: Keras, imdb dataset.

### Check out imdb_sentiment_classification_deep_learning.ipynb notebook in this repo to play around

### **Step 1:** import required python libraries


```python
import os
# Use Tensorflow backend
os.environ["KERAS_BACKEND"] = "tensorflow"

if os.environ["KERAS_BACKEND"] != "tensorflow" :
    #To use MKL 2018 with Theano you MUST set "MKL_THREADING_LAYER=GNU" in your environement.
    os.environ["MKL_THREADING_LAYER"] = "GNU"

from keras.datasets      import imdb
from keras.preprocessing import sequence
from keras               import callbacks
from keras.layers        import Embedding,LSTM,Dense
from keras.models        import Sequential,load_model
```

### **Step 2:** Load the IMDB dataset

IMDB reviews are already pre-processed in keras. Each review is stored as sequence of word indexes. Word is assigned an integer index based on its frequency. Higher the frequency lower the index. Index 0 is special index not assigned to particular word by to all unknown words.


```python
# Number of words
NUM_FEATURES = 25000

#indices returned by keras have <START> and <UNKNOWN> as indexes 1 and 2. (And it assumes you will use 0 for <PADDING>).
INDEX_FROM = 3 # This is anyway default parameter value. Explicitly mentioned here so that correcting word to index below is meaningful

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words = NUM_FEATURES, index_from=INDEX_FROM)

# build index to word  from word to index.
word_to_index = imdb.get_word_index()
# Adjust word_to_index considering actual word starts from INDEX_FROM
word_to_index = {k:(v+INDEX_FROM) for k,v in word_to_index.items()}
word_to_index["<PAD>"] = 0
word_to_index["<START>"] = 1
word_to_index["<UNK>"] = 2
index_to_word = dict(map(reversed,word_to_index.items()))
```


```python
print("Shape of x_train ", x_train.shape)
print("Shape of y_train ", y_train.shape)
print("Shape of x_test ", x_test.shape)
print("Shape of y_test ", y_test.shape)
```

    Shape of x_train  (25000,)
    Shape of y_train  (25000,)
    Shape of x_test  (25000,)
    Shape of y_test  (25000,)


### **Step 3:** Examine a move review by decoding from word indexes to english sentence. Also examine the sentiment label assigned to it - in training data.


```python
print("An example of review of length ", len(x_train[0]) ," words, encoded with sequence of word indexes\n", x_train[0])
```

    An example of review of length  218  words, encoded with sequence of word indexes
     [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 22665, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 21631, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 19193, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 10311, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 12118, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]



```python
print("Above review decoded into english words:\n", list(map(lambda x: index_to_word[x], x_train[0])))
```

    Above review decoded into english words:
     ['<START>', 'this', 'film', 'was', 'just', 'brilliant', 'casting', 'location', 'scenery', 'story', 'direction', "everyone's", 'really', 'suited', 'the', 'part', 'they', 'played', 'and', 'you', 'could', 'just', 'imagine', 'being', 'there', 'robert', "redford's", 'is', 'an', 'amazing', 'actor', 'and', 'now', 'the', 'same', 'being', 'director', "norman's", 'father', 'came', 'from', 'the', 'same', 'scottish', 'island', 'as', 'myself', 'so', 'i', 'loved', 'the', 'fact', 'there', 'was', 'a', 'real', 'connection', 'with', 'this', 'film', 'the', 'witty', 'remarks', 'throughout', 'the', 'film', 'were', 'great', 'it', 'was', 'just', 'brilliant', 'so', 'much', 'that', 'i', 'bought', 'the', 'film', 'as', 'soon', 'as', 'it', 'was', 'released', 'for', 'retail', 'and', 'would', 'recommend', 'it', 'to', 'everyone', 'to', 'watch', 'and', 'the', 'fly', 'fishing', 'was', 'amazing', 'really', 'cried', 'at', 'the', 'end', 'it', 'was', 'so', 'sad', 'and', 'you', 'know', 'what', 'they', 'say', 'if', 'you', 'cry', 'at', 'a', 'film', 'it', 'must', 'have', 'been', 'good', 'and', 'this', 'definitely', 'was', 'also', 'congratulations', 'to', 'the', 'two', 'little', "boy's", 'that', 'played', 'the', '<UNK>', 'of', 'norman', 'and', 'paul', 'they', 'were', 'just', 'brilliant', 'children', 'are', 'often', 'left', 'out', 'of', 'the', 'praising', 'list', 'i', 'think', 'because', 'the', 'stars', 'that', 'play', 'them', 'all', 'grown', 'up', 'are', 'such', 'a', 'big', 'profile', 'for', 'the', 'whole', 'film', 'but', 'these', 'children', 'are', 'amazing', 'and', 'should', 'be', 'praised', 'for', 'what', 'they', 'have', 'done', "don't", 'you', 'think', 'the', 'whole', 'story', 'was', 'so', 'lovely', 'because', 'it', 'was', 'true', 'and', 'was', "someone's", 'life', 'after', 'all', 'that', 'was', 'shared', 'with', 'us', 'all']



```python
print("Sentiment of above example review is ", ["Negative", "Positive"][y_train[0]])
```

    Sentiment of above example review is  Positive


### **Step 4:** Truncate all reviews to be of same length

Since keras or any neural nets does not take variable length inputs, truncate all reviews to be of same length


```python
MAX_LEN = 80 # 80 words in each review

# since ending part of review has mostly the concluding remarks of review, 
# we are truncating the beginning part of movie review and keeping last 80 words
x_train = sequence.pad_sequences(x_train, maxlen = MAX_LEN, padding='pre')
x_test = sequence.pad_sequences(x_test, maxlen = MAX_LEN, padding='pre')

```


```python
print("An example truncated review decoded into english words from training data:\n", list(map(lambda x: index_to_word[x], x_train[0])))
```

    An example truncated review decoded into english words from training data:
     ['that', 'played', 'the', '<UNK>', 'of', 'norman', 'and', 'paul', 'they', 'were', 'just', 'brilliant', 'children', 'are', 'often', 'left', 'out', 'of', 'the', 'praising', 'list', 'i', 'think', 'because', 'the', 'stars', 'that', 'play', 'them', 'all', 'grown', 'up', 'are', 'such', 'a', 'big', 'profile', 'for', 'the', 'whole', 'film', 'but', 'these', 'children', 'are', 'amazing', 'and', 'should', 'be', 'praised', 'for', 'what', 'they', 'have', 'done', "don't", 'you', 'think', 'the', 'whole', 'story', 'was', 'so', 'lovely', 'because', 'it', 'was', 'true', 'and', 'was', "someone's", 'life', 'after', 'all', 'that', 'was', 'shared', 'with', 'us', 'all']



```python
print("An example truncated review decoded into english words from test data:\n", list(map(lambda x: index_to_word[x], x_test[0])))
```

    An example truncated review decoded into english words from test data:
     ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<START>', 'please', 'give', 'this', 'one', 'a', 'miss', 'br', 'br', 'kristy', 'swanson', 'and', 'the', 'rest', 'of', 'the', 'cast', 'rendered', 'terrible', 'performances', 'the', 'show', 'is', 'flat', 'flat', 'flat', 'br', 'br', 'i', "don't", 'know', 'how', 'michael', 'madison', 'could', 'have', 'allowed', 'this', 'one', 'on', 'his', 'plate', 'he', 'almost', 'seemed', 'to', 'know', 'this', "wasn't", 'going', 'to', 'work', 'out', 'and', 'his', 'performance', 'was', 'quite', 'lacklustre', 'so', 'all', 'you', 'madison', 'fans', 'give', 'this', 'a', 'miss']


### **Step 5:** Building a 3-layer Keras Model for sentiment classification


```python
model = Sequential()
```

#### **Step 5a** First layer - Embedding layer - to learn a good representation of words


```python
# Create embedding of 128 dimensions for the NUM_FEATURES words
DIMENSIONS = 128
model.add(Embedding(input_dim=NUM_FEATURES, output_dim = DIMENSIONS))
```

#### **Step 5b** Second layer - LSTM - to learn key idea within the review


```python
model.add(LSTM(DIMENSIONS, dropout=0.2, recurrent_dropout=0.2))
```

#### **Step 5c** Third layer - Dense - transforms the idea to sentiment class


```python
model.add(Dense(1, activation='sigmoid'))
```

#### **Step 5d** Compile training model


```python
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
```

### **Step 6** Train the network

#### This step can be run multiple times - possibly it will increase accuracy if not flattened out already


```python
model_filename = "./imdb_lstm.h5"

if os.path.isfile(model_filename):
    print("Loading previously saved model:", model_filename)
    model = load_model(model_filename)
    print("Done loading model")

# Save model after each epoch.
checkpoint_callback = callbacks.ModelCheckpoint(filepath=model_filename, verbose = 1, save_best_only= True)

# Stop training when monitored quantity has stopped improving.
# stop training if the val_loss has reduced (i.e delta is < 0) for 2 (patience) consecutive times
earlystopping_callback = callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, verbose = 1, mode='auto')

model.fit(x=x_train, y=y_train, epochs=20, batch_size=64, 
          validation_data=(x_test, y_test),
          callbacks = [checkpoint_callback, earlystopping_callback])

loss,accuracy = model.evaluate(x=x_test, y=y_test,batch_size=64)
print("Loss:", loss)
print("Accuracy:", accuracy)
```

    Train on 25000 samples, validate on 25000 samples
    Epoch 1/20
    14464/25000 [================>.............] - ETA: 1:20 - loss: 0.0089 - acc: 0.9979

### **Step 7:** Train using TF-IDF and LogRegression using sklearn


```python
from sklearn.linear_model            import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics                 import accuracy_score,log_loss
from sklearn.pipeline                import Pipeline
```


```python
# convert integers in X_train as strings
x_train_s = [' '.join(map(str, row)) for row in x_train]
x_test_s  = [' '.join(map(str, row)) for row in x_test]

pipeline = Pipeline([['counter', TfidfVectorizer()],
                     ['classifier', LogisticRegression()]])

pipeline.fit(x_train_s, y_train)

y_predicted = pipeline.predict(x_test_s)

# predicted probablity estimates.
y_predicted_proba = pipeline.predict_proba(x_test_s)

print("Loss:", log_loss(y_test, y_predicted_proba))
print("Accuracy:", accuracy_score(y_test, y_predicted))

```

    Loss: 0.38315324286831653
    Accuracy: 0.84228


## Results

| Method       | Loss_Function      | Test Loss value| Test Accuracy|
|:-------------|:-------------------| -------------: |-------------:|
| LSTM         | binary_crossentropy|   0.379        |  83.6        |
| TF-IDF/LogReg| LogLoss            |   0.383        |  84.2        |


## Further Learning Resources
1. State-of-art Sentiment classifier - Unsupervised Sentiment Neuron - https://blog.openai.com/unsupervised-sentiment-neuron/
2. Understanding LSTM Networks https://colah.github.io/posts/2015-08-Understanding-LSTMs/
3. This repos is inspired by - GoDataDriven blogpost - https://blog.godatadriven.com/deep-learning-sentiment-classification
