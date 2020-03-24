import numpy as np

def similarity(a, b, metric):
    result = -100000
    if metric == 'cos':
        # a += 3
        # b += 3
        if (np.sum(a * a) == 0) or (np.sum(b * b) == 0):
            # print('The scalar has divided by zero.')
            result = -1
        else:
            result = (np.sum(a * b)) / ((np.sqrt(np.sum(a * a))) * (np.sqrt(np.sum(b * b))))

    elif metric == 'dotp':
        # a += 3
        # b += 3
        result = np.sum(a * b)
        # print('a*b',(a*b).shape)
    else:
        print('Invalid metric error')

    return result


def NearestNeighbor(target_vector, M, metric, mode):
    if mode == 'user':
        result = np.zeros(10916)
        for i in range(10916):
            # print(i)
            if i != 4321:
                result[i] = similarity(M[i, :].toarray(), target_vector.toarray(), metric)
                # print('M[i,:].toarray()', M[i, :].toarray().shape)
                # print('target_vector.toarray()', target_vector.toarray().shape)

            else:
                if metric == 'cos':
                    result[i] = -1
                elif metric == 'dotp':
                    result[i] = -10000
                else:
                    print('Invalid mode')

    elif mode == 'movie':
        result = np.zeros(5392)
        for j in range(5392):
            if j != 3:
                result[j] = similarity(M[:, j].toarray().T, target_vector.toarray().T, metric)
                # print('M[:,j].toarray()',M[:,j].toarray().T.shape)
                # print('target_vector.toarray()',target_vector.toarray().T.shape)

            else:
                if metric == 'cos':
                    result[j] = -1
                elif metric == 'dotp':
                    result[j] = -10000
                else:
                    print('Invalid mode')
    else:
        print('Invalid mode error')

    return result
