import numpy as np
from collections import OrderedDict

def parse_distro(path_topic_distro):
    distro_dict = OrderedDict()
    with open(path_topic_distro) as f:
        lines = f.readlines()
        # print('lines',lines)

        for line in lines:
            # print('line',line)
            user, query, distros = (line.strip().split(" ", 2))
            print('user',user,'query',query)
            distros = distros.split(" ")

            for i in range(12):
                distros[i] = float(distros[i].split(":")[1])
            # distro_dict["%s-%s" % (user, query)] = distros
            distro_dict["".join([str(user),'-',str(query)])] = distros
            # print(sorted(distro_dict.items()))

        print(distro_dict)
        # print(distro_dict.keys())
        # print(distro_dict.values())
    return distro_dict

def weighted_sum_r(r_topic, distro_dict, query_num):
  distro = np.asarray(list(distro_dict.values())) # 38 x 12
  weighted_r = np.zeros(81433)

  for i in range(12):
    # print(i,'-th weighted sum')
    weighted_r += r_topic[:,i] * distro[query_num,i]
  # distro[query_num,:].transpose() # 81433x12 * 12x1 = 81433x1
  return weighted_r


def parse_indri(filename):
    doc_relscore_list = []
    with open(filename) as f:
        for line in f:
            doc = int(line.strip().split(" ")[2])
            rel_score = float(line.strip().split(" ")[4])
            doc_relscore_list.append([doc, rel_score])

    doc_relscore_list = np.asarray(doc_relscore_list)
    return doc_relscore_list

def get_fname(distro_dict):
  fname=list(distro_dict.keys())
  for i in range(len(fname)):
    fname[i]="".join(['C:\\Users\\chlee\\PycharmProjects\\ML_Text_Mining_HW1\\hw1handout\\hw1-handout\\data\\indri-lists\\',fname[i],'.results.txt'])
  return fname


def get_NS(r, doc_relscore_list):
    NS = []

    for i in doc_relscore_list[:, 0]:
        NS.append((i, r[int(i - 1)]))

    NS = sorted(NS, key=lambda x: (-x[1], x[0]))
    for i in range(len(NS)):
        NS[i] = list(NS[i])

    NS = np.asarray(NS)

    return NS


def get_WS(r, doc_relscore_list, w1, w2):
    WS = []
    for i in doc_relscore_list[:, 0]:
        WS.append((i, w1 * r[int(i - 1)] + w2 * np.asscalar(doc_relscore_list[doc_relscore_list[:, 0] == i, 1])))
    WS = sorted(WS, key=lambda x: (-x[1], x[0]))

    for i in range(len(WS)):
        WS[i] = list(WS[i])
    WS = np.asarray(WS)

    return WS


def get_CM(r, doc_relscore_list, w1, w2):
    CM = []
    for i in doc_relscore_list[:, 0]:
        CM.append(
            (i, w1 * 10 * r[int(i - 1)] ** 2 + w2 * np.asscalar(doc_relscore_list[doc_relscore_list[:, 0] == i, 1])))
    CM = sorted(CM, key=lambda x: (-x[1], x[0]))

    for i in range(len(CM)):
        CM[i] = list(CM[i])
    CM = np.asarray(CM)

    return CM

def get_doc_relscore_list(fname, query):
  doc_relscore_list = parse_indri(fname[query])
  doc_relscore_list = np.asarray(doc_relscore_list)
  return doc_relscore_list
