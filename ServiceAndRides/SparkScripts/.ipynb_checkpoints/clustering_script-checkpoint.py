import sys
import numpy as np
import pandas as pd
from math import sqrt
from datetime import datetime
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.sql.session import SparkSession
from pyspark.ml.clustering import KMeans, BisectingKMeans

# The imports, above, allow us to access SparkML features
def vector_from_inputs(r):
  return (r["int64_field_0"], Vectors.dense(
                      float(r["latitude"]),
                      float(r["longitude"])))

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))


# Use Cloud Dataprocs automatically propagated configurations to get
# the Cloud Storage bucket and Google Cloud Platform project for this
# cluster.
sc = SparkContext()
spark = SparkSession(sc)
bucket = sc._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
project = sc._jsc.hadoopConfiguration().get('fs.gs.project.id')

# Set an input directory for reading data from Bigquery.
todays_date = datetime.strftime(datetime.today(), "%Y-%m-%d-%H-%M-%S")
input_directory = "gs://{}/tmp/clustering-{}".format(bucket, todays_date)


# Set the configuration for importing data from BigQuery.
# Specifically, make sure to set the project ID and bucket for Cloud Dataproc,
# and the project ID, dataset, and table names for BigQuery.
#summerai:nyc_mv_collisions_processed.collision_data
conf = {
    # Input Parameters
    "mapred.bq.project.id": project,
    "mapred.bq.gcs.bucket": bucket,
    "mapred.bq.temp.gcs.path": input_directory,
    "mapred.bq.input.project.id": project,
    "mapred.bq.input.dataset.id": "nyc_mv_collisions_processed",
    "mapred.bq.input.table.id": "collision_data",
}

# Read the data from BigQuery into Spark as an RDD.
table_data = spark.sparkContext.newAPIHadoopRDD(
    "com.google.cloud.hadoop.io.bigquery.JsonTextBigQueryInputFormat",
    "org.apache.hadoop.io.LongWritable",
    "com.google.gson.JsonObject",
    conf=conf)


#Data needed to be converted to a Dataframe to use the pyspark.ml API
#So, we call data from GCS, extra JSON Strings and Create a Temp View
#TempView is Queried using SQL Syntax and converted to suitable DF for further fitting
# Extract the JSON strings from the RDD.
table_json = table_data.map(lambda x: x[1])
# Load the JSON strings as a Spark Dataframe.
data = spark.read.json(table_json)
# Create a view so that Spark SQL queries can be run against the data.
data.createOrReplaceTempView("clustering")
# As a precaution, run a query in Spark SQL to ensure no NULL values exist.
sql_query = """
SELECT *
from clustering
where int64_field_0 is not null #Unique Identifier
and latitude is not null
and longitude is not null
"""
clean_data = spark.sql(sql_query)
# Create an input DataFrame for Spark ML using the above function.
training_data = clean_data.rdd.map(vector_from_inputs).toDF(["label", "features"])
training_data.cache()


print('*******************************************************')
#print(training_data.rdd.getNumPartitions)
training_data.repartition(1000)

#Training Model
k = 10
kmeans = BisectingKMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(training_data)
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

#Assign Clusters to events
transformed = model.transform(training_data).select('label', 'prediction')
rows = transformed.collect()
print(rows[:3])
df_pred = spark.createDataFrame(rows)
df_pred.repartition(1) \
    .write \
    .mode("overwrite") \
    .format("csv") \
    .option("header", "true") \
    .save("gs://graphx-usage/clusters")
# Save and load model
#clusters.save(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")
#sameModel = KMeansModel.load(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")
#spark-submit --jars=gs://hadoop-lib/bigquery/bigquery-connector-hadoop2-latest.jar k3.py




