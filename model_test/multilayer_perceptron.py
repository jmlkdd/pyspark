# coding=utf-8

import sys
# 由于PYTHONPATH找不到pyspark包，这里手动添加路径
sys.path.append('/usr/local/spark-2.1.1-bin-hadoop2.7/python')

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = SparkContext('local', 'logistic_regression')
spark = SparkSession(sc)

# Load training data
data = spark.read.format("libsvm").load("../data/mllib/sample_multiclass_classification_data.txt")

# Split the data into train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [4, 5, 4, 3]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

# train the model
model = trainer.fit(train)

# compute accuracy on the test set
result = model.transform(test)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
