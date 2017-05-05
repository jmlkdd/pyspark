# coding=utf-8

import sys
# 由于PYTHONPATH找不到pyspark包，这里手动添加路径
sys.path.append('/usr/local/spark-2.1.1-bin-hadoop2.7/python')

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = SparkContext('local', 'logistic_regression')
spark = SparkSession(sc)


# load data file.
inputData = spark.read.format("libsvm").load("../data/mllib/sample_multiclass_classification_data.txt")

# generate the train/test split.
(train, test) = inputData.randomSplit([0.8, 0.2])

# instantiate the base classifier.
lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)

# instantiate the One Vs Rest Classifier.
ovr = OneVsRest(classifier=lr)

# train the multiclass model.
ovrModel = ovr.fit(train)

# score the model on test data.
predictions = ovrModel.transform(test)

# obtain evaluator.
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

# compute the classification error on test data.
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))
