from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans
from numpy import array

sc = SparkContext()

#12 records with height, weight data
data = array([185,72, 170,56, 168,60, 179,68, 182,72, 188,77, 180,71, 180,70, 183,84, 180,88, 180,67, 177,76]).reshape(12,2)

#Generate Kmeans
model = KMeans.train(sc.parallelize(data), 2, runs=50, initializationMode="random")

model.save(sc, "savedModelDir")
