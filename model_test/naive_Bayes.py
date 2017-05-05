# coding=utf-8

import sys
# 由于PYTHONPATH找不到pyspark包，这里手动添加路径
sys.path.append('/usr/local/spark-2.1.1-bin-hadoop2.7/python')

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = SparkContext('local', 'logistic_regression')
spark = SparkSession(sc)

# Load training data
data = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")

# Split the data into train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
model = nb.fit(train)

# select example rows to display.
predictions = model.transform(test)
predictions.show()

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))
