from pmf import PMF
import numpy as np
import utils
import sys
print(sys.path)
path_data = 'C:\\Users\\chlee\\PycharmProjects\\ML_Text_Mining_HW2\\hw2-handout\\data\\train.csv'
data = utils.load_raw_review_data(path_data)

print('Total number of movies :', len(np.unique(data[0])))
print('Total number of users :', len(np.unique(data[1])))

movies = np.asarray(list(data[0]))
# print(movies)
users = np.asarray(list(data[1]))
ratings = np.asarray(list(data[2]))

print('len(movies)', len(movies))
print('len(users)', len(users))
print('len(ratings)', len(ratings))

print('max(data[0])', max(data[0]))
print('max(data[1])', max(data[1]))

alpha = 0.005
lambda_u = 0.001
lambda_v = 0.001
batch_size = 10
num_iterations = 150
num_features = 30
pmf = PMF(num_features, max(data[1])+1, max(data[0]+1))
pmf.train(users, movies, ratings, alpha, lambda_u, lambda_v, batch_size, num_iterations)

path_dev_data = 'C:\\Users\\chlee\\PycharmProjects\\ML_Text_Mining_HW2\\hw2-handout\\data\\dev.csv'
import pandas as pd
dev_data = pd.read_csv(path_dev_data, header=None, delimiter=',')
dev_data = np.asarray(dev_data)

dev_movies = dev_data[:,0]
dev_users = dev_data[:,1]
# ratings = np.asarray(list(dev_data[2]))

for i in range(len(dev_movies)):
    predicted = pmf.predict(dev_users[i], dev_movies[i])
    print(predicted)

