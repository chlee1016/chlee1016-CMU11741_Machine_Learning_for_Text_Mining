##########################################
# import libs
##########################################
import time
from PR_Modules import *
import pandas as pd
import numpy as np
import time
from scipy.sparse import coo_matrix
from numpy import linalg as LA

##########################################
# Load transition matrix data
##########################################

path_data = 'C:\\Users\\chlee\\PycharmProjects\\ML_Text_Mining_HW1\\hw1handout\\hw1-handout\\data\\transition.txt'
data_transition = pd.read_csv(path_data, header=None, delimiter=' ')
data_transition = np.asarray(data_transition)
# column of data_transition composed of column i j k, and k=1 means that there is a link from document i and document j.
print(data_transition)
print(type(data_transition))
print(data_transition.shape)

# maximum value of row
print('Dimension of row : ',max(data_transition[:,0]))

# maximum value of column
print('Dimension of column : ',max(data_transition[:,1]))
##########################################
# Preprocess & Normalize the transition matrix
##########################################

# Make transition matrix into squared matrix

row = data_transition[:,0]
col = data_transition[:,1]
val = data_transition[:,2]

print('Length of row vector',len(row))
print('Length of col vector',len(col))
print('Length of val vector',len(val))

M = coo_matrix((val, (row-1, col-1)), shape = (81433,81433))

n_i = M.sum(axis=1)
n_i = np.asarray(n_i)
n_i = np.divide(1, n_i, where=n_i!=0)
K = np.squeeze((n_i==0))

# if n_i > 0
#  if there is a link from i to j and j != i, then M_ij = 1/n_i
#  if there is no link from i to j or if j = i, then M_ij = 0

n_i=n_i.reshape((81433,))
row_N = np.arange(81433)
col_N = np.arange(81433)

M_A = coo_matrix((n_i, (row_N, col_N)), shape = (81433,81433))
M_A = M_A * M

##########################################
# Define initial values
##########################################
n = 81433

p_0 = (np.ones(n)/n ).T
r = (np.ones(n)* 1/n ).T

alpha = 0.2

r_distance = 10
error_threshold = 10**(-8)
iter_count = 0

##########################################
# Calculate the r vector
##########################################

while error_threshold < r_distance:
    r_k_1 = r
    r = (1 - alpha) * M_A.T * r + (1 - alpha) * (1 / n) * sum(r[K]) * np.ones(n).T + alpha * p_0
    r_k = r
    r_distance = LA.norm(r_k - r_k_1, 2)
    print(r_distance)
    iter_count += 1

print('sum of r vector:', sum(r))  # 1.000000000000743
print('maximum score of r :', max(r))
print('argmax of r :', np.argmax(r))
print('iteration count :', iter_count)

f = open("C:\\Users\\chlee\\PycharmProjects\\ML_Text_Mining_HW1\\hw1handout\\hw1-handout\\data\\GPR.txt",'w')
for i in range(len(r)):
    f.write(" ".join([str(i+1), str(r[i]), '\n']))
f.close()



##########################################
# Load doc-topic data
##########################################
path_doc_topic = 'C:\\Users\\chlee\\PycharmProjects\\ML_Text_Mining_HW1\\hw1handout\\hw1-handout\\data\\doc_topics.txt'
data_doc_topic = pd.read_csv(path_doc_topic, header=None, delimiter=' ')
print(data_doc_topic)

##########################################
# Define initial values
##########################################
n = 81433

p_0 = (np.ones(n)/n ).T

alpha = 0.8
beta = 0.15
gamma = 0.05

error_threshold = 10**(-8)

##########################################
# Make a list of topic correspond to the docs.
##########################################
import sys

np.set_printoptions(threshold=sys.maxsize)
data_doc_topic = np.asarray(data_doc_topic)
num_topic = 12

p_t = np.zeros((num_topic, n), dtype=bool)
print(p_t.shape)
for i in range(num_topic):
    Topic_Doc = data_doc_topic[data_doc_topic[:, 1] == (i + 1), 0]
    Topic_Doc = np.sort(Topic_Doc)
    n_t = len(Topic_Doc)

    for j in Topic_Doc[:]:
        p_t[i, (j - 1)] = True

row_N = np.arange(12)
col_N = np.arange(12)
Norm_factor = coo_matrix(((1 / np.sum(p_t, axis=1)), (row_N, col_N)), shape=(12, 12))
print(Norm_factor)
p_t = Norm_factor * p_t

##########################################
# Calculate the r vector
##########################################
r_topic = []

for i in range(12):

    r = (np.ones(n) * 1 / n).T
    r_distance = 10
    iter_count = 0

    print('iteration :', i)
    while error_threshold < r_distance:
        r_k_1 = r
        r = alpha * M_A.T * r + alpha * (1 / n) * sum(r[K]) * np.ones(n).T + beta * p_t[i, :] + gamma * p_0
        r_k = r

        r_distance = LA.norm((r_k - r_k_1), 2)
        # print('r_distance', r_distance)
        iter_count += 1

    print(iter_count)

    r_topic.append(r)

r_topic = np.asarray(r_topic)
r_topic = r_topic.T
print(r_topic.shape)

print('sum of r vector:', np.sum(r))
print('maximum score of r :', max(r))


##########################################
# PTSPR : Load user-topic-distro
##########################################
path_user_topic_distro = 'C:\\Users\\chlee\\PycharmProjects\\ML_Text_Mining_HW1\\hw1handout\\hw1-handout\\data\\user-topic-distro.txt'
distro_dict = parse_distro(path_user_topic_distro)
distro_dict_keys = np.asarray(list(distro_dict.keys())) # 38 x 12
fname = get_fname(distro_dict)



##########################################
# Save QTSPR-U2Q1
##########################################

f = open("C:\\Users\\chlee\\PycharmProjects\\ML_Text_Mining_HW1\\hw1handout\\hw1-handout\\data\\PTSPR-U2Q1.txt",'w')
weighted_r = weighted_sum_r(r_topic,distro_dict,12)
for i in range(len(weighted_r)):
    f.write(" ".join([str(i+1), str(weighted_r[i]), '\n']))
f.close()

##########################################
# PTSPR - NS
##########################################
start = time.time()

f = open("C:\\Users\\chlee\\PycharmProjects\\ML_Text_Mining_HW1\\hw1handout\\hw1-handout\\data\\PTSPR-NS.txt",'w')
for i in range(38):
  weighted_r = weighted_sum_r(r_topic,distro_dict,i)
  doc_relscore_list = get_doc_relscore_list(fname,i)
  NS = get_NS(weighted_r,doc_relscore_list) # 1 column : doc_id (start from 1)
  for j in range(len(NS[:,0])):
    f.write(" ".join([distro_dict_keys[i], 'Q0', str(int(NS[j,0])), str((j+1)), str(NS[j,1]), 'indri','run-1','\n']))

f.close()
print("PTSPR-NS time :", time.time() - start)


##########################################
# PTSPR - WS
##########################################
start = time.time()

f = open("C:\\Users\\chlee\\PycharmProjects\\ML_Text_Mining_HW1\\hw1handout\\hw1-handout\\data\\PTSPR-WS.txt",'w')
for i in range(38):
  weighted_r = weighted_sum_r(r_topic,distro_dict,i)
  doc_relscore_list = get_doc_relscore_list(fname,i)
  WS = get_WS(weighted_r, doc_relscore_list, 1000, 1) # 1 column : doc_id (start from 1), 2 column : ws score
  for j in range(len(WS[:,0])):
    f.write(" ".join([distro_dict_keys[i], 'Q0', str(int(WS[j,0])), str((j+1)), str(round(WS[j,1],5)), 'indri','run-1','\n']))

f.close()
print("PTSPR-WS time :", time.time() - start)


##########################################
# PTSPR - CM
##########################################
start = time.time()

f = open("C:\\Users\\chlee\\PycharmProjects\\ML_Text_Mining_HW1\\hw1handout\\hw1-handout\\data\\PTSPR-CM.txt",'w')
for i in range(38):
  weighted_r = weighted_sum_r(r_topic,distro_dict,i)
  doc_relscore_list = get_doc_relscore_list(fname,i)
  CM = get_CM(weighted_r, doc_relscore_list, 1000, 1) # 1 column : doc_id (start from 1), 2 column : ws score
  for j in range(len(CM[:,0])):
    f.write(" ".join([distro_dict_keys[i], 'Q0', str(int(CM[j,0])), str((j+1)), str(round(CM[j,1],5)), 'indri','run-1','\n']))

f.close()

print('file writing is finished.')

print("PTSPR-CM time :", time.time() - start)
