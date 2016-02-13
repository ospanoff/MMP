import sklearn
from sklearn import neighbors
import numpy as np

def get_mnist_predictions(train, train_answer, test):
    neighborsNum = 5
    blockSize = 3500
    name = "brute"
    met = "cosine"
    eps = 0.01
    
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors = neighborsNum,\
                                              algorithm = name, metric = met).fit(train)
    tmp1 = np.empty((test.shape[0], neighborsNum), dtype = np.int32)
    tmp2 = np.empty((test.shape[0], neighborsNum), dtype = np.int32)
    for i in range(0, test.shape[0], blockSize):
        indxL = i
        indxR = indxL + blockSize
        tmp = nbrs.kneighbors(test[indxL : indxR])
        tmp1[indxL : indxR, :] = tmp[0]
        tmp2[indxL : indxR, :] = tmp[1]

    dist = tmp1
    labels = train_answer[tmp2]

    reslbls = np.zeros(labels.shape[0])

    for i in range(labels.shape[0]):
        sum = np.zeros(np.max(train_answer))
        for j in range(labels[i].size):
            sum[labels[i][j]] += 1 / (dist[i][j] + eps)
            reslbls[i] = np.argmax(sum)
    
    return reslbls