import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
np.random.seed(7)
data_path = "C:/Users/chlee/PycharmProjects/ML_Text_Mining_HW3/hw3handout/hw3-handout/data/data/"

# vocab_size = 21959
vocab_size = 10000
max_review_length = 3817
embedding_vector_length = 100

X_train = np.load(data_path + 'train_inputs.npy')
y_train = np.load(data_path + 'train_labels.npy')
X_test = np.load(data_path + 'test_inputs.npy')
y_test = np.load(data_path + 'test_labels.npy')

###################################
print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)
###################################


###################################
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D


model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_review_length))
# model.add(Dropout(0.2))
model.add(Conv1D(100, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), shuffle=True, epochs=5, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

###################################
for key in hist.history.keys():
    print(key)

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([-0.2, 1.2])

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylim([-0.2, 1.2])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

