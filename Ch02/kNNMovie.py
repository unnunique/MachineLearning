'''
reated on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: lidiliang
'''
from numpy import *
import operator
## movie dataSets
def createMovieDataSets():
    dataSets = array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
    lables = ['A', 'A', 'A', 'B', 'B', 'B']
    return dataSets, lables

def classify0(inX, dataSet, lables, k):
    dataSetSize = dataSet.shape[0]
    diffMatV0 = tile(inX, (dataSetSize, 1)) - dataSet
    diffMatV1 = diffMatV0**2
    distancesV0 = diffMatV1.sum(axis=1)
    distancesV1 = distancesV0**0.5
    sortedDistancesIndexes = distancesV1.argsort()
    classCount={}
    for i in range(k):
        voteLable = lables[sortedDistancesIndexes[i]]
        classCount[voteLable] = classCount.get(voteLable, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


