import numpy as np
import pandas as pd
import re
from glob import glob
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords



data_path = "C:/Users/chlee/PycharmProjects/ML_Text_Mining_HW3/hw3handout/hw3-handout/data/data/"

train_list = glob(data_path + "train/*/*")
test_list = glob(data_path + "test/*/*")
print('The length of train_list', len(train_list))
print('The length of test_list', len(test_list))

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


train_sentence_list = get_sentence_list(train_list)
train_data = pd.DataFrame({'sentence' : train_sentence_list, 'label' : [0]*1000 + [1]*1000})

test_sentence_list = get_sentence_list(test_list)
test_data = pd.DataFrame({'sentence' : test_sentence_list, 'label' : [0]*1000 + [1]*1000})




clean_train_sentences = []
for sentence in train_data['sentence']:
    clean_train_sentences.append(preprocessing(sentence, remove_stopwords=True))

clean_test_sentences = []
for sentence in test_data['sentence']:
    clean_test_sentences.append(preprocessing(sentence, remove_stopwords=True))

# print('clean_train_sentences[0]', clean_train_sentences[0])
# print('clean_test_sentences[0]', clean_test_sentences[0])

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(clean_train_sentences)
train_text_sequences = tokenizer.texts_to_sequences(clean_train_sentences)
test_text_sequences = tokenizer.texts_to_sequences(clean_test_sentences)

print('Maximum of train_text_sequences', max(train_text_sequences[3]))
print('Maximum of test_text_sequences', max(test_text_sequences[3]))

clean_train_df = pd.DataFrame({'sentence': clean_train_sentences, 'label': train_data['label']})
clean_test_df = pd.DataFrame({'sentence': clean_test_sentences, 'label': test_data['label']})


###################################
# Generate and save vocab
word_vocab = tokenizer.word_index.items()
print('total number of words in vocab', len(word_vocab))
print(word_vocab)

data_configs = {}
data_configs['vocab'] = word_vocab
data_configs['vocab_size'] = len(word_vocab) + 1
print(data_configs['vocab'])
###################################

MAX_SEQUENCE_LENGTH = 3817
train_inputs = pad_sequences(train_text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
test_inputs = pad_sequences(test_text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

train_labels = np.array(train_data['label'])
print('Shape of train data: ', train_inputs.shape)
print('Shape of train label: ', train_labels.shape)
np.save(data_path + 'train_inputs', train_inputs)
np.save(data_path + 'train_labels', train_labels)

test_labels = np.array(test_data['label'])
print('Shape of test data: ', test_inputs.shape)
print('Shape of test label: ', test_labels.shape)
np.save(data_path + 'test_inputs', test_inputs)
np.save(data_path + 'test_labels', test_labels)





# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
#
# def Rmv_stopwords(word_tokens):
#     stop_words = set(stopwords.words('english'))
#     result = []
#     for w in word_tokens:
#         if w not in stop_words:
#             result.append(w)
#     return result
#
#
#
# ###################################
# # Make vocab
# voca = set([])
# corpus = []
# doc_len = []
#
# for i in range(len(train_list)):
#     train = open(train_list[i], 'r')
#     train_lines = train.readlines()
#     sentence = train_lines[0]
#     corpus.append(sentence)
#     # Tokenize the sentence
#     word_tokens = word_tokenize(sentence)
#     doc_len.append(len(word_tokens))
#     # Remove stopwords
#     result = Rmv_stopwords(word_tokens)
#     # result = word_tokens
#
#     voca.update(result)
#     train.close()
#
# ###################################
# # Basic statistics
# print('The total number of unique words in T', len(voca))
# print('The total number of training examples in T', len(train_list))
# print('The ratio of positive examples to negative examples in T',
#       len(glob(data_path + "train/positive/*")) / len(glob(data_path + "train/negative/*")))
# print('The average length of document in T', sum(doc_len)/len(doc_len))
# print('The max length of document in T', max(doc_len))

###################################
# # Make vacab by using top 10000 words
# from collections import Counter
# total_sentence = ""
#
# for i in range(len(train_list)):
#     train = open(train_list[i], 'r')
#     train_lines = train.readlines()
#     sentence = train_lines[0]
#     total_sentence += sentence
#     train.close()
#
# total_corpus = word_tokenize(total_sentence)
# total_corpus_stwrds = Rmv_stopwords(total_corpus)
# print(len(list(set(total_corpus_stwrds)))) # 25578
# count = Counter(total_corpus_stwrds)
#
# tag_count = []
# tags = []
# for n, c in count.most_common(10000):
#     dics = {'tag':n, 'count': c}
#     tags.append(n)
#     tag_count.append(c)
# print('tags', len(tags))
# print('tag_count', len(tag_count))
#
# print(tags)
# print(tag_count)
#
# ###################################


