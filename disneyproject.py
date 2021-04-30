# %%

from __future__ import print_function

import re
import sys
from re import sub

import nltk
import numpy as np
from pyspark import SparkContext
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, col, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.linalg import Vectors

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)

    # spark = SparkSession.builder.appName('python word count').getOrCreate()
    sc = SparkContext(appName="A7")

    lines = sc.textFile(sys.argv[1], 1)
    stripped = lines.map(lambda x: sub("<[^>]+>", "", x))

    nltk.download('stopwords')

    # Define a list of stop words or use default list

    spark = SparkSession.builder.appName('python').getOrCreate()
    spark.catalog.clearCache()
    sc = spark.sparkContext
    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)

    file = sc.textFile(sys.argv[1], 1)
    print(file)


    def freqArray(listOfIndices, numberofwords):
        returnVal = np.zeros(5000)
        for index in listOfIndices:
            returnVal[index] = returnVal[index] + 1
        returnVal = np.divide(returnVal, numberofwords)
        return returnVal


    def buildArray(listOfIndices):
        returnVal = np.zeros(5000)
        for index in listOfIndices:
            returnVal[index] = returnVal[index] + 1
        mysum = np.sum(returnVal)
        returnVal = np.divide(returnVal, mysum)
        return returnVal


    schema = StructType([
        StructField("Review_ID", IntegerType()),
        StructField("Rating", IntegerType(), True),
        StructField("Year_Month", StringType(), True),
        StructField("Reviewer_Location", StringType(), True),
        StructField("Review_Text", StringType(), True),
        StructField("Branch", StringType(), True)
    ])

    dfWithSchema = spark.read.schema(schema).csv('DisneylandReviews.csv')

    print(dfWithSchema.printSchema())
    dfWithSchema = dfWithSchema.select(concat(col("Review_ID"), lit("-"), col("Branch")), col("Review_Text"))
    print(dfWithSchema.show())
    dfWithSchema = dfWithSchema.withColumnRenamed("concat(Review_ID, -, Branch)", "Branch")
    print(dfWithSchema.show())
    print(1)
    dfWithSchema = dfWithSchema.select(col("Branch"), col("Review_Text"))
    print(dfWithSchema.show())
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
    a = StopWordsRemover(inputCol="Review_Text", outputCol="Filtered_Reviews", stopWords=stopwords)
    b = a.transform(dfWithSchema)
    df = b.select(col("Branch"), col("Filtered_Reviews"))
    d_keyAndListOfWords = df.rdd.map(tuple)
    print(d_keyAndListOfWords.take(3))

    allWords = d_keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))
    print(allWords.take(1))
    allCounts = allWords.reduceByKey(lambda a, b: a + b)
    print(allCounts.take(1))
    # # # Get the top 20,000 words in a local array in a sorted format based on frequency
    # # # If you want to run it on your laptio, it may a longer time for top 20k words.
    topWords = allCounts.top(5000, lambda x: x[1])
    print("Top Words in Corpus:", allCounts.top(10, key=lambda x: x[1]))
    topWordsK = sc.parallelize(range(5000))
    print(topWordsK.take(10))
    dictionary = topWordsK.map(lambda x: (topWords[x][0], x))
    print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ", dictionary.top(20, lambda x: x[1]))
    allWordsWithDocID = d_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
    allDictionaryWords = dictionary.join(allWordsWithDocID)
    justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))
    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
    allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))
    zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0], np.clip(np.multiply(x[1], 9e9), 0, 1)))
    dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]
    multiplier = np.full(5000, numberOfDocs)
    idfArray = np.log(np.divide(np.full(5000, numberOfDocs), dfArray))
    allDocsAsNumpyArrays = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))


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
    new_RDD = myRDD.map(lambda x: (convert(x[0]), x[1]))
    # new_RDD = new_RDD.map(lambda x: LabeledPoint(x[0], x[1]))
    # new_RDD.cache()

    print(new_RDD.count())
    trainingData, testData = new_RDD.randomSplit([.5, .5])
    print(testData.take(5))
    new_RDD = new_RDD.map(lambda x: (x[0], Vectors.dense([float(i) for i in x[1]])))
    data = spark.createDataFrame(new_RDD)
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="_1", featuresCol="_2", numTrees=10)
    model = rf.fit(trainingData)
    # Make predictions.
    predictions = model.transform(testData)

#### GET MODEL RESULTS
    # Select example rows to display.
    predictions.select("_1", "_2", "prediction", "probability").show(5)
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="_1", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))
    evaluate_importance = model.featureImportances
    print(evaluate_importance)
    importance = model.featureImportances
    # summarize feature importance
    features = []
    values = []
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
        features.append(i)
        values.append(v)
    final = sorted(zip(features, values), key=lambda x: x[1], reverse=True)
    print(final)

    top_words = final[0:50]
    print(top_words)


    words = dictionary.collect()

    print(words)

    for word in top_words:
        get_index = int(word[0])
        print(words[get_index][0])

    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    eval_accuracy = MulticlassClassificationEvaluator(labelCol="_1", predictionCol="prediction", metricName="accuracy")
    eval_precision = MulticlassClassificationEvaluator(labelCol="_1", predictionCol="prediction", metricName="precisionByLabel")
    eval_recall = MulticlassClassificationEvaluator(labelCol="_1", predictionCol="prediction", metricName="recallByLabel")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="_1", predictionCol="prediction", metricName="f1")
    eval_auc = BinaryClassificationEvaluator(labelCol="_1", rawPredictionCol="prediction")

    accuracy = eval_accuracy.evaluate(predictions)
    precision = eval_precision.evaluate(predictions)
    recall = eval_recall.evaluate(predictions)
    f1score = eval_f1.evaluate(predictions)
    auc = eval_accuracy.evaluate(predictions)

    print('accuracy', accuracy)
    print('precision', precision)
    print('recall', recall)
    print('flscore', f1score)
    print('auc', auc)

    #calculate results per label
    actual_predicted = predictions.select("_1", "prediction")
    by_label = actual_predicted.groupBy("_1", "prediction").count()
    print(by_label.show())

    # labels = data.map(lambda lp: lp.label).distinct().collect()
    # for label in sorted(labels):
    #     print("Class %s precision = %s" % (label, eval_precision.precision(label)))
    #     print("Class %s recall = %s" % (label, eval_recall.recall(label)))