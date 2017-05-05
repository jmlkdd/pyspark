# coding=utf-8

import sys
sys.path.append('/usr/local/spark-2.1.1-bin-hadoop2.7/python')

import numpy as np
import scipy.sparse as sps
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession


if __name__ == '__main__':
    # conf = SparkConf().setAppName('pyspark_test').setMaster('spark://10.10.10.196:7077')
    # sc = SparkContext(conf)
    # sc = SparkContext("spark://10.10.10.196:7077", "NetworkWordCount")
    sc = SparkContext('local', 'test')
    spark = SparkSession(sc)
    dv1 = np.array([1, 0, 3])
    dv2 = [1, 0, 3]
    sv1 = Vectors.sparse(3, [0, 2], [1, 3])
    sv2 = sps.csc_matrix((np.array([1, 3]), np.array([0, 2]), np.array([0, 2])), shape=(3, 1))
    print dv1
    print dv2
    print sv1
    print sv2

    pos = LabeledPoint(1.0, [1.0, 0.0, 3.0])
    print pos.label, pos.features

    neg = LabeledPoint(0.0, SparseVector(3, [0, 2], [1.0, 3.0]))

    print neg.features