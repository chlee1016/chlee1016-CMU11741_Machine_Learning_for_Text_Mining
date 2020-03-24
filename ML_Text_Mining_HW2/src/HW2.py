'''
##########################################
# Machine Learning for Text Mining
# Homework 2
# Name : Changhyun, Lee
# Date : 23th, Feb, 2020
##########################################
'''
# Import library
import numpy as np
import utils
from Recommendation_modules import*
from scipy.sparse import csr_matrix

##########################################
# Load the data

path_data = 'C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\data\\train.csv'
data = utils.load_raw_review_data(path_data)

print('Total number of movies :', len(np.unique(data[0])))
print('Total number of users :', len(np.unique(data[1])))

movie_id = np.asarray(list(data[0]))
user_id = np.asarray(list(data[1]))
rate_id = np.asarray(list(data[2]))

# print('The number of times any movie was rated 1 : ',len(np.unique(movie_id[rate_id==1])))
# print('The number of times any movie was rated 3 : ',len(np.unique(movie_id[rate_id==3])))
# print('The number of times any movie was rated 5 : ',len(np.unique(movie_id[rate_id==5])))

print('The number of times any movie was rated 1 : ', len(movie_id[rate_id == 1]))
print('The number of times any movie was rated 3 : ', len(movie_id[rate_id == 3]))
print('The number of times any movie was rated 5 : ', len(movie_id[rate_id == 5]))

print('The average movie rating across all users and movies : ', np.average(rate_id))

##########################################
# 1.1 Basic statistics
# For user ID 4321
print('The number of movies rated : ', np.sum(user_id == 4321))
print('The number of times the user gave a \'1\' rating', sum(rate_id[user_id == 4321] == 1))
print('The number of times the user gave a \'3\' rating', sum(rate_id[user_id == 4321] == 3))
print('The number of times the user gave a \'5\' rating', sum(rate_id[user_id == 4321] == 5))
print('The average movie rating for this user', np.average(rate_id[user_id == 4321]))

##########################################
# For movie ID 3
print('The number of movies rated : ', np.sum(movie_id == 3))
print('The number of times the movies gave a \'1\' rating', sum(rate_id[movie_id == 3] == 1))
print('The number of times the movies gave a \'3\' rating', sum(rate_id[movie_id == 3] == 3))
print('The number of times the movies gave a \'5\' rating', sum(rate_id[movie_id == 3] == 5))
print('The average movie rating for this movie', np.average(rate_id[movie_id == 3]))

##########################################
# 1.2 Nearest Neighbors

# import sys
# np.set_printoptions(threshold=sys.maxsize)

review_data = utils.load_review_data_matrix(path_data)
print(review_data.rows)
print(review_data.cols)
print(review_data.data)

print(max(review_data.rows))
print(max(review_data.cols))
print(max(review_data.data))

# user-movie data
M = csr_matrix((review_data.data, (review_data.rows, review_data.cols)), shape=(10916, 5392))




##########################################
import sys

np.set_printoptions(threshold=sys.maxsize)

'''
# For user ID 4321
# Top K NNs of user 4321 in terms of dot product similarity
K=5
result = NearestNeighbor(M[4321,:], M, 'dotp', 'user')
print('Top 5 NNs of user 4321 in terms of dot product similarity : \n', np.where(result > np.sort(result)[-(K+1)]))

# Top 5 NNs of user 4321 in terms of cosine similarity
result = NearestNeighbor(M[4321,:], M, 'cos', 'user')
print('Top 5 NNs of user 4321 in terms of cosine similarity : \n', np.where(result > np.sort(result)[-(K+1)]))

##########################################
# For movie ID 5
# Top 5 NNs of movie 5 in terms of dot product similarity
result = NearestNeighbor(M[:,3], M, 'dotp', 'movie')
print('Top 5 NNs of movie 3 in terms of dot product similarity : \n', np.where(result > np.sort(result)[-(K+1)]))

# Top 5 NNs of movie 5 in terms of cosine similarity
result = NearestNeighbor(M[:,3], M, 'cos', 'movie')
print('Top 5 NNs of movie 3 in terms of cosine similarity : \n', np.where(result > np.sort(result)[-(K+1)]))
'''

# M_center = csr_matrix((review_data.data, (review_data.rows, review_data.cols)), shape=(10916, 5392))

# print('review_data.data[review_data.row==0]', review_data.data[review_data.rows==0])
# rate = review_data.data
# for i in range(10916):
#     for j in range(len(review_data.data)):
#         rate[j] = rate[j] - M_mean[i]
# rate = np.zeros(len(review_data.data))
# print(M_mean[review_data.rows].shape)



path_dev_data = 'C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\data\\dev.csv'
import pandas as pd
dev_data = pd.read_csv(path_dev_data, header=None, delimiter=',')
dev_data = np.asarray(dev_data)

path_test_data = 'C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\data\\test.csv'
import pandas as pd
test_data = pd.read_csv(path_test_data, header=None, delimiter=',')
test_data = np.asarray(test_data)

A = M * M.T
A.setdiag(-10000)
user_norm = np.zeros(10916)
for i in range(10916):
    user_norm[i] = np.sqrt(np.sum(np.squeeze(M[i, :].toarray() * M[i, :].toarray())))



B = M.T * M
B.setdiag(-10000)
movie_norm = np.zeros(5392)
for i in range(5392):
    movie_norm[i] = np.sqrt(np.sum(np.squeeze(M[:, i].toarray() * M[:, i].toarray())))



def user_user_similarity(movie,user,K,metric, rating_method):
    rating = 0



    if metric == 'cos':

        result = np.squeeze(A[user, :].toarray())
        ###########################################
        # normalization for cosine similarity
        norm = np.divide(1,user_norm, where=user_norm != 0)
        if user_norm[user] != 0:
            result = result * ((1/user_norm[user]) * norm)
        ###########################################

        threshold = np.sort(result)[-(K + 1)]
        top_k_id = np.where(result >= threshold)[0]

        score = result[top_k_id]
        while len(top_k_id) > K:
            top_k_id = np.delete(top_k_id, np.argmin(score))

        # print('top_k_id', top_k_id)

        if rating_method == 'mean':
            for i in top_k_id:
                rating = rating + M[i, movie]
            rating = rating / K

        elif rating_method == 'weighted':
            for i in top_k_id:
                rating = rating + (result[i] * M[i, movie])
            # print('result[top_k_id', result[top_k_id])
            rating = rating / np.sum(np.abs(result[top_k_id]))




    elif metric == 'dotp':
        result = np.squeeze(A[user,:].toarray())
        threshold = np.sort(result)[-(K + 1)]
        # print('threshold', threshold)

        top_k_id = np.where(result >= threshold)[0]
        # print('top_k_id',top_k_id)

        score = result[top_k_id]
        while len(top_k_id) > K:
            top_k_id = np.delete(top_k_id, np.argmin(score))

        # print('top_k_id', top_k_id)

        for i in top_k_id:
            rating = rating + M[i, movie]
        rating = rating / K

    else :
        print('Invalid metric')

    return rating


#########################################
def movie_movie_similarity(movie,user,K,metric, rating_method):
    rating = 0



    if metric == 'cos':

        result = np.squeeze(B[:, movie].toarray())
        ###########################################
        # normalization for cosine similarity
        norm = np.divide(1,movie_norm, where=movie_norm != 0)
        if movie_norm[movie] != 0:
            result = result * ((1/movie_norm[movie]) * norm)
        ###########################################

        threshold = np.sort(result)[-(K + 1)]
        top_k_id = np.where(result >= threshold)[0]

        score = result[top_k_id]
        while len(top_k_id) > K:
            top_k_id = np.delete(top_k_id, np.argmin(score))

        # print('top_k_id', top_k_id)

        if rating_method == 'mean':
            for i in top_k_id:
                rating = rating + M[user, i]
            rating = rating / K

        elif rating_method == 'weighted':
            for i in top_k_id:
                rating = rating + (result[i] * M[user, i])
            # print('result[top_k_id', result[top_k_id])
            rating = rating / np.sum(np.abs(result[top_k_id]))




    elif metric == 'dotp':
        result = np.squeeze(B[:,movie].toarray())
        threshold = np.sort(result)[-(K + 1)]
        # print('threshold', threshold)

        top_k_id = np.where(result >= threshold)[0]
        # print('top_k_id',top_k_id)

        score = result[top_k_id]
        while len(top_k_id) > K:
            top_k_id = np.delete(top_k_id, np.argmin(score))

        # print('top_k_id', top_k_id)

        for i in top_k_id:
            rating = rating + M[user, i]
        rating = rating / K

    else :
        print('Invalid metric')

    return rating

#########################################


def knn(data, K, metric, rating_method):

    predicted_list = np.zeros(len(data))
    for i in range(len(data)):
        # print(i, 'th iteration')
        predicted_list[i] = user_user_similarity(data[i, 0], data[i, 1], K, metric, rating_method)

    predicted_list = predicted_list + 3
    print(K, ' ', metric, ' ', rating_method, ' finished')
    return predicted_list
#########################################

def knn_movie(data, K, metric, rating_method):

    predicted_list = np.zeros(len(data))
    for i in range(len(data)):
        # print(i, 'th iteration')
        predicted_list[i] = movie_movie_similarity(data[i, 0], data[i, 1], K, metric, rating_method)

    predicted_list = predicted_list + 3
    print(K, ' ', metric, ' ', rating_method, ' finished')
    return predicted_list

import time

##########################################



##########################################
# User-User similarity
##########################################
# 1. Mean, Dotp, K=10

start_time = time.time()
predicted_list = knn(dev_data, 10, 'dotp', 'mean')
print('Mean, Dotp, K=10 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pred_dev_mean_dotp_10.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
# 2. Mean, Dotp, K=100

start_time = time.time()
predicted_list = knn(dev_data, 100, 'dotp', 'mean')
print('Mean, Dotp, K=100 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pred_dev_mean_dotp_100.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
# 3. Mean, Dotp, K=500

start_time = time.time()
predicted_list = knn(dev_data, 500, 'dotp', 'mean')
print('Mean, Dotp, K=500 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pred_dev_mean_dotp_500.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
#4. Mean, Cos, K=10

start_time = time.time()
predicted_list = knn(dev_data, 10, 'cos', 'mean')
print('Mean, cos, K=10 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pred_dev_mean_cos_10.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()


##########################################
#5. Mean, Cos, K=100

start_time = time.time()
predicted_list = knn(dev_data, 100, 'cos', 'mean')
print('Mean, cos, K=100 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pred_dev_mean_cos_100.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
#6. Mean, Cos, K=500

start_time = time.time()
predicted_list = knn(dev_data, 500, 'cos', 'mean')
print('Mean, cos, K=500 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pred_dev_mean_cos_500.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
#7. Weighted, Cos, K=10

start_time = time.time()
predicted_list = knn(dev_data, 10, 'cos', 'weighted')
print('Weighted, cos, K=10 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pred_dev_weighted_cos_10.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
#8. Weighted, Cos, K=100

start_time = time.time()
predicted_list = knn(dev_data, 100, 'cos', 'weighted')
print('Weighted, cos, K=100 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pred_dev_weighted_cos_100.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
#9. Weighted, Cos, K=500

start_time = time.time()
predicted_list = knn(dev_data, 500, 'cos', 'weighted')
print('Weighted, cos, K=500 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pred_dev_weighted_cos_500.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()


#########################################
# Movie-Movie similarity
#########################################
# 1. Mean, Dotp, K=10

start_time = time.time()
predicted_list = knn_movie(dev_data, 10, 'dotp', 'mean')
print('Mean, Dotp, K=10 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\movie_pred_dev_mean_dotp_10.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
# 2. Mean, Dotp, K=100

start_time = time.time()
predicted_list = knn_movie(dev_data, 100, 'dotp', 'mean')
print('Mean, Dotp, K=100 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\movie_pred_dev_mean_dotp_100.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
# 3. Mean, Dotp, K=500

start_time = time.time()
predicted_list = knn_movie(dev_data, 500, 'dotp', 'mean')
print('Mean, Dotp, K=500 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\movie_pred_dev_mean_dotp_500.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
# 4. Mean, Cos, K=10

start_time = time.time()
predicted_list = knn_movie(dev_data, 10, 'cos', 'mean')
print('Mean, Cos, K=10 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\movie_pred_dev_mean_cos_10.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
# 5. Mean, Cos, K=100

start_time = time.time()
predicted_list = knn_movie(dev_data, 100, 'cos', 'mean')
print('Mean, Cos, K=100 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\movie_pred_dev_mean_cos_100.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
# 6. Mean, Cos, K=500

start_time = time.time()
predicted_list = knn_movie(dev_data, 500, 'cos', 'mean')
print('Mean, Cos, K=500 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\movie_pred_dev_mean_cos_500.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
# 7. Weighted, Cos, K=10

start_time = time.time()
predicted_list = knn_movie(dev_data, 10, 'cos', 'weighted')
print('Mean, Cos, K=10 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\movie_pred_dev_weighted_cos_10.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
# 8. Weighted, Cos, K=100

start_time = time.time()
predicted_list = knn_movie(dev_data, 100, 'cos', 'weighted')
print('Mean, Cos, K=100 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\movie_pred_dev_weighted_cos_100.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

#########################################
# 9. Weighted, Cos, K=500

start_time = time.time()
predicted_list = knn_movie(dev_data, 500, 'cos', 'weighted')
print('Mean, Cos, K=500 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\movie_pred_dev_weighted_cos_500.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()


##########################################
# PCC similarity
##########################################


M_sum = np.sum(M.toarray(),axis=1)
M_num = np.zeros(10916)
# for i in range(10916):
#     # M_num[i] = np.sum(np.squeeze(M[i,:].toarray()) != 0)
#     # print(np.sum(np.squeeze(M[i, :].toarray()) != 0))
#
#     M_num[i] = np.sum(review_data.rows == i)
# print('M_sum', M_sum)
# print('M_num', M_num)
# M_mean = np.divide(M_sum, M_num, where= M_num != 0)



M_mean = M_sum/5392
print('M_mean', M_mean)
M = csr_matrix((review_data.data - M_mean[review_data.rows], (review_data.rows, review_data.cols)), shape=(10916, 5392))
A = M * M.T
A.setdiag(-10000)
user_norm = np.zeros(10916)
for i in range(10916):
    user_norm[i] = np.sqrt(np.sum(np.squeeze(M[i, :].toarray() * M[i, :].toarray())))
print('Matrix M is redefined')


##########################################
# 1. Mean, Dotp, K=10

start_time = time.time()
predicted_list = knn(dev_data, 10, 'dotp', 'mean')
print('Mean, Dotp, K=10 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pcc_pred_dev_mean_dotp_10.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
# 2. Mean, Dotp, K=100

start_time = time.time()
predicted_list = knn(dev_data, 100, 'dotp', 'mean')
print('Mean, Dotp, K=100 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pcc_pred_dev_mean_dotp_100.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
# 3. Mean, Dotp, K=500

start_time = time.time()
predicted_list = knn(dev_data, 500, 'dotp', 'mean')
print('Mean, Dotp, K=500 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pcc_pred_dev_mean_dotp_500.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
#4. Mean, Cos, K=10

start_time = time.time()
predicted_list = knn(dev_data, 10, 'cos', 'mean')
print('Mean, cos, K=10 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pcc_pred_dev_mean_cos_10.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()


##########################################
#5. Mean, Cos, K=100

start_time = time.time()
predicted_list = knn(dev_data, 100, 'cos', 'mean')
print('Mean, cos, K=100 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pcc_pred_dev_mean_cos_100.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
#6. Mean, Cos, K=500

start_time = time.time()
predicted_list = knn(dev_data, 500, 'cos', 'mean')
print('Mean, cos, K=500 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pcc_pred_dev_mean_cos_500.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
#7. Weighted, Cos, K=10

start_time = time.time()
predicted_list = knn(dev_data, 10, 'cos', 'weighted')
print('Weighted, cos, K=10 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pcc_pred_dev_weighted_cos_10.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
#8. Weighted, Cos, K=100

start_time = time.time()
predicted_list = knn(dev_data, 100, 'cos', 'weighted')
print('Weighted, cos, K=100 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pcc_pred_dev_weighted_cos_100.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()

##########################################
#9. Weighted, Cos, K=500

start_time = time.time()
predicted_list = knn(dev_data, 500, 'cos', 'weighted')
print('Weighted, cos, K=500 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\pcc_pred_dev_weighted_cos_500.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()




##########################################
# Predictions for test
##########################################
# Mean, Dotp, K=10

start_time = time.time()
predicted_list = knn(test_data, 10, 'dotp', 'mean')
print('Mean, Dotp, K=10 running time : ', time.time()-start_time)
predicted_list = np.nan_to_num(predicted_list,nan=3)
f = open("C:\\Users\\chlee\\PycharmProjects\\ML_hw2\\hw2-handout\\eval\\test-predictions.txt",'w')
for i in range(len(predicted_list)):
    f.write(" ".join([str(predicted_list[i]), "\n"]))
f.close()
