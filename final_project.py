from __future__ import print_function

import re
import nltk
import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest

nltk.download('stopwords')

# Define a list of stop words or use default list

spark = SparkSession.builder.appName('python word count').getOrCreate()
sc = spark.sparkContext
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

def freqArray (listOfIndices, numberofwords):
	returnVal = np.zeros (20000)
	for index in listOfIndices:
		returnVal[index] = returnVal[index] + 1
	returnVal = np.divide(returnVal, numberofwords)
	return returnVal


def buildArray(listOfIndices):
	returnVal = np.zeros(20000)
	for index in listOfIndices:
		returnVal[index] = returnVal[index] + 1
	mysum = np.sum(returnVal)
	returnVal = np.divide(returnVal, mysum)
	return returnVal

file = pd.read_csv("/Users/shezalpadani/Downloads/DisneylandReviews.csv")  # assuming the file contains a header
df = sqlContext.createDataFrame(file)
dfWithSchema = spark.createDataFrame(file).toDF("Review_ID", "Rating", "Year_Month", "Reviewer_Location", "Review_Text", "Branch")
dfWithSchema = dfWithSchema.select(col("Branch"),col("Review_Text"))
rdd = dfWithSchema.rdd.map(tuple)
numberOfDocs = rdd.count()
print(numberOfDocs)
print(rdd.take(5))
regex = re.compile('[^a-zA-Z]')
d_keyAndListOfWords = rdd.map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()))
print(d_keyAndListOfWords.take(4))
remover = StopWordsRemover()
stopwords = remover.getStopWords()
stopwords = stopwords + ['hong', 'kong', 'california', 'paris']
print(stopwords)

dfWithSchema = spark.createDataFrame(d_keyAndListOfWords).toDF("branch", "Review_Text")
a = StopWordsRemover(inputCol="Review_Text", outputCol="Filtered_Reviews" ,stopWords=stopwords)
b = a.transform(dfWithSchema)
df = b.select(col("Branch"),col("Filtered_Reviews"))
d_keyAndListOfWords = df.rdd.map(tuple)
print(d_keyAndListOfWords.take(3))

allWords = d_keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))
print(allWords.take(1))
allCounts = allWords.reduceByKey(lambda a, b: a + b)
print(allCounts.take(1))
# # # Get the top 20,000 words in a local array in a sorted format based on frequency
# # # If you want to run it on your laptio, it may a longer time for top 20k words.
topWords = allCounts.top(20000, lambda x : x[1])
print("Top Words in Corpus:", allCounts.top(10, key=lambda x: x[1]))
topWordsK = sc.parallelize(range(20000))
dictionary = topWordsK.map (lambda x : (topWords[x][0], x))
print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ", dictionary.top(20, lambda x : x[1]))
allWordsWithDocID = d_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
allDictionaryWords = dictionary.join(allWordsWithDocID)
justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))
allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))
print(allDocsAsNumpyArrays.take(10))
zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0], np.clip (np.multiply (x[1], 9e9), 0, 1)))
dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]
multiplier = np.full(20000, numberOfDocs)
idfArray = np.log(np.divide(np.full(20000, numberOfDocs), dfArray))
allDocsAsNumpyArrays = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))
print(allDocsAsNumpyArrays.take(10))

def convert(value):
	if 'HongKong' in value:
		return 0.0
	elif 'California' in value:
		return 1.0
	else:
		return 2.0


# convert AU to 1 and wiki docs to 0
myRDD = allDocsAsNumpyArrays
test = myRDD.map(lambda x: x[0])
print(test.take(1000))
new_RDD = myRDD.map(lambda x: (convert(x[0]), x[1]))
new_RDD = new_RDD.map(lambda x: LabeledPoint(x[0], x[1]))


# print(new_RDD.count())
# # Split the data into training and test sets (30% held out for testing)
# trainingData, testData = new_RDD.randomSplit([.5, .5])
# print(trainingData.count())
# print(testData.count())
#
# import psutl
# # Train a RandomForest model.
# #  Empty categoricalFeaturesInfo indicates all features are continuous.
# #  Note: Use larger numTrees in practice.
# #  Setting featureSubsetStrategy="auto" lets the algorithm choose.
# model = RandomForest.trainClassifier(trainingData, numClasses=3, categoricalFeaturesInfo={},
# 									 numTrees=10, featureSubsetStrategy="auto",
#                                      impurity='gini', maxDepth=5, maxBins=32)
#
# # Evaluate model on test instances and compute test error
# predictions = model.predict(testData.map(lambda x: x.features))
# labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
# print(labelsAndPredictions)
# testErr = labelsAndPredictions.filter(
#     lambda lp: lp[0] != lp[1]).count() / float(testData.count())
# print('Test Error = ' + str(testErr))
# print('Learned classification forest model:')
# print(model.toDebugString())