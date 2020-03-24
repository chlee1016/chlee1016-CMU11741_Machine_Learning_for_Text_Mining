import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import time
from glob import glob
from nltk.corpus import stopwords
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D


def preprocessing(sentence, remove_stopwords = False):

    review_text = re.sub("[^a-zA-Z]", " ", sentence)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        clean_review = ' '.join(words)
    else:
        clean_review = ' '.join(words)

    return clean_review


def get_sentence_list(data_path_list):
    sentence_list = []
    for i in range(len(data_path_list)):
        data = open(data_path_list[i], 'r')
        data_lines = data.readlines()
        sentence = data_lines[0]
        sentence_list.append(sentence)
        data.close()
    return sentence_list


def get_sentence_data(train_path_list, test_path_list):
    train_sentence_list = get_sentence_list(train_path_list)
    train_data = pd.DataFrame({'sentence' : train_sentence_list, 'label' : [0]*1000 + [1]*1000})

    test_sentence_list = get_sentence_list(test_path_list)
    test_data = pd.DataFrame({'sentence' : test_sentence_list, 'label' : [0]*1000 + [1]*1000})


    clean_train_sentences = []
    for sentence in train_data['sentence']:
        clean_train_sentences.append(preprocessing(sentence, remove_stopwords=True))

    clean_test_sentences = []
    for sentence in test_data['sentence']:
        clean_test_sentences.append(preprocessing(sentence, remove_stopwords=True))

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(clean_train_sentences)
    train_text_sequences = tokenizer.texts_to_sequences(clean_train_sentences)
    test_text_sequences = tokenizer.texts_to_sequences(clean_test_sentences)

    MAX_SEQUENCE_LENGTH = 3817

    X_train = pad_sequences(train_text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    X_test = pad_sequences(test_text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # clean_train_df = pd.DataFrame({'sentence': clean_train_sentences, 'label': train_data['label']})
    # clean_test_df = pd.DataFrame({'sentence': clean_test_sentences, 'label': test_data['label']})

    y_train = np.array(train_data['label'])
    print('Shape of X_train: ', X_train.shape)
    print('Shape of y_train: ', y_train.shape)
    np.save(data_path + 'X_train', X_train)
    np.save(data_path + 'y_train', y_train)

    y_test = np.array(test_data['label'])
    print('Shape of X_test: ', X_test.shape)
    print('Shape of y_test: ', y_test.shape)
    np.save(data_path + 'X_test', X_test)
    np.save(data_path + 'y_test', y_test)
    print('finished saving data')
    ###################################
    return tokenizer


def get_pre_trained_embedding(embedding_vector_path, train_path_list):
    embedding_dict = dict()
    f = open(embedding_vector_path, encoding='utf8')
    for line in f:
        word_vector = line.split()
        word = word_vector[0]
        word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
        embedding_dict[word] = word_vector_arr
    f.close()
    print("total %s of embedding vectors are exist in GloVe data." % len(embedding_dict))

    ###################################
    embedding_matrix = np.zeros((vocab_size, 100))

    train_sentence_list = get_sentence_list(train_path_list)
    train_data = pd.DataFrame({'sentence': train_sentence_list, 'label': [0] * 1000 + [1] * 1000})
    clean_train_sentences = []
    for sentence in train_data['sentence']:
        clean_train_sentences.append(preprocessing(sentence, remove_stopwords=True))
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(clean_train_sentences)

    # Confirm the vocab is composed by the frequency.
    print('tokenizer.word_counts', sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True))
    print('tokenizer.word_index', tokenizer.word_index)

    for word, i in tokenizer.word_index.items():
        # extract the word(key) from training vocab.
        if i == vocab_size:
            break
        temp = embedding_dict.get(word)
        # save the embedding vector which is from GloVe to temp.
        if temp is not None:
            embedding_matrix[i] = temp
            # map the GloVe embedding vector into embedding_matrix.

    print('finished generating embedding matrix')
    print('shape of embedding_matrix', np.shape(embedding_matrix))

    return embedding_matrix


def get_model_cnn():
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_review_length, trainable=True))
    model.add(Conv1D(100, 3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def get_model_cnn_pre_trained(embedding_matrix):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_review_length,
                        weights=[embedding_matrix], trainable=False))
    model.add(Conv1D(100, 3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def visualize(hist):
    plt.figure(1)

    plt.subplot(211)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(212)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



###################################
# main

data_path = "C:/Users/chlee/PycharmProjects/ML_Text_Mining_HW3/hw3handout/hw3-handout/data/data/"

train_list = glob(data_path + "train/*/*")
test_list = glob(data_path + "test/*/*")
print('The length of train_list', len(train_list))
print('The length of test_list', len(test_list))

get_sentence_data(train_list, test_list)

np.random.seed(7)
vocab_size = 10000
max_review_length = 3817
embedding_vector_length = 100

X_train = np.load(data_path + 'X_train.npy')
y_train = np.load(data_path + 'y_train.npy')
X_test = np.load(data_path + 'X_test.npy')
y_test = np.load(data_path + 'y_test.npy')

###################################
print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)
###################################


model = get_model_cnn()
start_time = time.time()
hist_cnn = model.fit(X_train, y_train, validation_data=(X_test, y_test), shuffle=True, epochs=5, batch_size=16)
print("training time for cnn:", time.time()-start_time)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
visualize(hist_cnn)



embedding_vector_path = 'C:/Users/chlee/PycharmProjects/ML_Text_Mining_HW3/glove.6B/glove.6B.100d.txt'
embedding_matrix = get_pre_trained_embedding(embedding_vector_path, train_list)
model = get_model_cnn_pre_trained(embedding_matrix)
start_time = time.time()
hist_cnn_pre_trained = model.fit(X_train, y_train, validation_data=(X_test, y_test), shuffle=True, epochs=5, batch_size=16)
print("training time for cnn with pre-trained:", time.time()-start_time)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
visualize(hist_cnn_pre_trained)




