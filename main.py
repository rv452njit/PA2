import sys
import os
import boto3
import s3fs
import pandas as pd
import numpy as np
import pyspark
import torch
from pyspark.mllib.linalg import Vectors
from pyspark.ml.regression import RandomForestRegressor
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.tree import RandomForest,RandomForestModel
from pyspark.mllib.evaluation import MulticlassMetrics
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.shell import spark
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#other way to read files from bucket
#df = pd.read_csv('s3://rv452-bkt/input/TrainingDataset.csv')
#valdf = pd.read_csv('s3://rv452-bkt/input/ValidationDataset.csv')

#sparkDF=spark.createDataFrame(df) 
#valSparkDF = spark.createDataFrame(valdf)

S3_DATA_SOURCE_PATH_VALID = 's3://rv452-emr/datasource/ValidationDataset.csv'
S3_DATA_SOURCE_PATH = 's3://rv452-emr/datasource/TrainingDataset.csv'
S3_DATA_OUTPUT_PATH = 's3://rv452-emr/output

def main():
    spark = SparkSession.builder.appName('Predictions').getOrCreate()
    all_data = spark.read.csv(S3_DATA_SOURCE_PATH, header = True, inferSchema='true', sep=';')
    valid_data = spark.read.csv(S3_DATA_SOURCE_PATH_VALID, header = True, inferSchema='true', sep=';')

    print(all_data.count())

    featureColumns = [c for c in all_data.columns if c != 'quality']
    columns = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

    transformed_df= all_data.rdd.map(lambda row: (
                                                float((row)[0].split(";")[0]),
                                                float((row)[0].split(";")[1]),
                                                float((row)[0].split(";")[2]),
                                                float((row)[0].split(";")[3]),
                                                float((row)[0].split(";")[4]),
                                                float((row)[0].split(";")[5]),
                                                float((row)[0].split(";")[6]),
                                                float((row)[0].split(";")[7]),
                                                float((row)[0].split(";")[8]),
                                                float((row)[0].split(";")[9]),
                                                float((row)[0].split(";")[10]),
                                                float((row)[0].split(";")[11])
                                                ))

    valtransformed_df= valid_data.rdd.map(lambda row: (
                                                float((row)[0].split(";")[0]),
                                                float((row)[0].split(";")[1]),
                                                float((row)[0].split(";")[2]),
                                                float((row)[0].split(";")[3]),
                                                float((row)[0].split(";")[4]),
                                                float((row)[0].split(";")[5]),
                                                float((row)[0].split(";")[6]),
                                                float((row)[0].split(";")[7]),
                                                float((row)[0].split(";")[8]),
                                                float((row)[0].split(";")[9]),
                                                float((row)[0].split(";")[10]),
                                                float((row)[0].split(";")[11])
                                                ))


    newtransformed_df = transformed_df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))
    newValtransformed_df = valtransformed_df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))

    model = RandomForest.trainClassifier(newtransformed_df,numClasses=10,categoricalFeaturesInfo={}, numTrees=50, maxBins=64, maxDepth=20, seed=33)
    #model.save(sc,'s3://rv452-bkt/output/model_created.model')

    #predictions
    predictions = model.predict(newValtransformed_df.map(lambda x: x.features))

    labelsAndPredictions = newValtransformed_df.map(lambda lp: lp.label).zip(predictions)

    labelsAndPredictions_df = labelsAndPredictions.toDF()

    print(labelsAndPredictions_df)
    #cpnverting rdd ==> spark dataframe ==> pandas dataframe 
    labelpred = labelsAndPredictions.toDF(["quality", "Prediction"])
    #labelpred.show()
    labelpred_df = labelpred.toPandas()

    #Calculating the F1score
    F1score = f1_score(labelpred_df['quality'], labelpred_df['Prediction'], average='micro')
    print("F1- score: ", F1score)
    print(confusion_matrix(labelpred_df['quality'],labelpred_df['Prediction']))
    print(classification_report(labelpred_df['quality'],labelpred_df['Prediction']))
    print("Accuracy" , accuracy_score(labelpred_df['quality'], labelpred_df['Prediction']))

    #calculating the test error
    testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(newValtransformed_df.count())    
    print('Test Error = ' + str(testErr))

    #Other ways of pushing models to the s3 bucket
    from boto3.s3.transfer import S3Transfer
    #have all the variables populated which are required below
    client = boto3.client('s3', aws_access_key_id="AKIA2L3FCO3X37SKDWGE",aws_secret_access_key="Nk-L88bnz4X0sBfJlpkwdBVFm3sew307p2lG+GGN")
    transfer = S3Transfer(client)
    filepath = 'models.model'
    bucket_name = 'rv452-bkt'
    folder_name = 'output'
    filename= 'models.model'
    transfer.upload_file(filepath, bucket_name, folder_name+"/"+filename)

    labelpred_df.write.mode('overwrite').parquet(S3_DATA_OUTPUT_PATH)
    print('Suceeded')
if __name__== '__main__':
    main()