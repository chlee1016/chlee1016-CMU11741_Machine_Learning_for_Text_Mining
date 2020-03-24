import pandas as pd
import numpy as np
import time

# import sys
# np.set_printoptions(threshold=sys.maxsize)

path_data = 'C:\\Users\\chlee\\PycharmProjects\\ML_Text_Mining_HW1\\hw1handout\\hw1-handout\\data\\transition.txt'

#Explicitly pass header=0 to be able to replace existing names
data_transition = pd.read_csv(path_data, header=None, delimiter=' ')

data_transition = np.asarray(data_transition)
print(type(data_transition))
print(data_transition.shape)
# maximum value of row
print('Dimension of row : ',max(data_transition[:,0]))

# maximum value of column
print('Dimension of column : ',max(data_transition[:,1]))
np.array(len(data_transition))

from scipy.sparse import coo_matrix
#Make transition matrix into squared matrix

row = data_transition[:,0]
col = data_transition[:,1]
val = data_transition[:,2]

print('Length of row vector',len(row))
print('Length of col vector',len(col))
print('Length of val vector',len(val))


M = coo_matrix((val, (row-1, col-1)), shape = (81433,81433))
# print(M)
# M_T = coo_matrix((val, (col-1, row-1)), shape = (81433,81433))

##########################################
# Make an n_i vector
n_i = M.sum(axis=1)
n_i = np.asarray(n_i)

##########################################
# Check sum-zero rows
# import sys
# np.set_printoptions(threshold=sys.maxsize)
# print(n_i==0)


##########################################
# if n_i > 0
#  if there is a link from i to j and j != i, then M_ij = 1/n_i
#  if there is no link from i to j or if j = i, then M_ij = 0
# M.setdiag(0)

K = np.squeeze((n_i==0))

print(sum(K))

# for i in range(81433):
#   if n_i[i] != 0 :
#     n_i[i] = 1/n_i[i]
n_i = np.divide(1, n_i, where=n_i!=0)
# print(len(n_i))

n_i=n_i.reshape((81433,))
row_N = np.arange(81433)
col_N = np.arange(81433)

print(n_i)
M_A = coo_matrix((n_i, (row_N, col_N)), shape = (81433,81433))
print(M_A)
M_A = M_A * M
# print(M_A)

# Set initial values of alpha, n, p_0
alpha = 0.2
n = max(row[:]) # n = 81433
print('alpha : ', alpha)
print('n : ', n)

p_0 = (np.ones(n)/n ).T
print(p_0)

# Set initial values of r as 1/n for each element
r = (np.ones(n)* 1/n ).T
# r=np.arange(10)

# for i in range(10):
#   r = (1-alpha)*M_A.T*r + (1-alpha)*(1/n)*sum( r[K] ) * np.ones(n).T + alpha*p_0

print('sum of r vector:', sum(r)) # 1.000000000000743
print(max(r))
print(r)
print(np.argmax(r))

path_doc_topic = 'C:\\Users\\chlee\\PycharmProjects\\ML_Text_Mining_HW1\\hw1handout\\hw1-handout\\data\\doc_topics.txt'

#Explicitly pass header=0 to be able to replace existing names
data_doc_topic = pd.read_csv(path_doc_topic, header=None, delimiter=' ')
#
print(data_doc_topic)

import sys
np.set_printoptions(threshold=sys.maxsize)
data_doc_topic = np.asarray(data_doc_topic)

Topic1_Doc = data_doc_topic[data_doc_topic[:,1]==1,0]
Topic1_Doc = np.sort(Topic1_Doc)

print(type(Topic1_Doc))
print(Topic1_Doc)
n_t = len(Topic1_Doc)
# print(Topic1_Doc)

p_t = np.zeros(n,dtype=bool)
# print(len(C))
for i in Topic1_Doc[:]:
  print(i)
  p_t[i]=True
p_t = p_t/sum(p_t)
print(p_t.shape)

alpha = 0.8
beta = 0.15
gamma = 0.05
for i in range(10):
  r= alpha * M_A.T*r + alpha*(1/n)*sum( r[K] ) * np.ones(n).T + beta*p_t + gamma * p_0
# print(r)
print(sum(r))

##########################################
def parse_distro(filename):
  """

  Parse the personalization/query distribution information from file

  :param filename: the text file

  :return: a dictionary in the form of {user-query: [p1, p2, ...]}

  """

  distro_dict = dict(list)

  with open(filename) as f:

    for line in f:

      # split one line into 3 parts

      user, query, distros = (line.strip().split(" ", 2))

      for topic_prob in distros.split(" "):
        # add every probability of topic to the dictionary

        distro_dict["%s-%s" % (user, query)].append(float(topic_prob.split(":")[1]))

  return distro_dict


def parse_indri(filename):
  """

  Parse the indri doc search-relevance scores from the a indri file

  :param filename: the indri file

  :return: list of (doc, score) tuples

  """

  res_list = []

  with open(filename) as f:
    for line in f:
      doc = int(line.strip().split(" ")[2])

      score = float(line.strip().split(" ")[4])

      res_list.append((doc, score))

  return res_list

