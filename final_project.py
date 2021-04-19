from __future__ import print_function

import re
import sys

import nltk
import numpy as np
from operator import add
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import DenseVector
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import array, avg, col
import pandas as pd
import sys
from operator import add
from pyspark import SparkContext
import re
import sys
import numpy as np
from operator import add
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import DenseVector
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import array, avg, col
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
import string
import re
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Tokenizer

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
zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0], np.clip (np.multiply (x[1], 9e9), 0, 1)))
dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]
print(dfArray.take(5))
multiplier = np.full(20000, numberOfDocs)
idfArray = np.log(np.divide(np.full(20000, numberOfDocs), dfArray))
print(idfArray.take(5))
allDocsAsNumpyArrays = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))
print(allDocsAsNumpyArrays.take(2))
# allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))
# print(allDocsAsNumpyArrays.take(30))