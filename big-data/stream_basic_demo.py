from data.dataset import CompanyDataset
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.streaming import StreamingContext
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier

dataset = CompanyDataset(0.2, 0.2, strategy='tibi_mean')

spark = SparkSession.builder.master("local[*]").appName("project").getOrCreate()
sc = spark.sparkContext.getOrCreate()


def convert_to_spark_df(data, input_cols):
    dataset_spark = sc.parallelize(data)
    df = dataset_spark.map(lambda x: x.tolist()).toDF(input_cols + ["label"])
    df = df.withColumn('label', df['label'].cast(IntegerType()))
    return df


def assemble_dataframe(df, input_cols):
    assembler = VectorAssembler(
        inputCols=dataset.columns,
        outputCol="features")

    output = assembler.transform(df)
    output = output.select(["features", "label"])
    return output


train = spark.read.csv("train.csv", header=True, inferSchema=True)
test = spark.read.csv("streaming_test_data.csv", header=True, inferSchema=True)
train_assembled = assemble_dataframe(train, dataset.columns)
test_assembled = assemble_dataframe(test, dataset.columns)


tree = DecisionTreeClassifier()
tree_model = tree.fit(train_assembled)

ssc = StreamingContext(sc, 10)
lines = ssc.socketTextStream("localhost", 9999)
words = lines.flatMap(lambda line: line.split(" "))


def mapRdd(currentRdd):
    if currentRdd is not None:
        print(currentRdd)
        currentRddData = currentRdd.collect()
        print(currentRddData)
        print(type(currentRddData))
        for i in currentRddData:
            if "csv" in i:
                test = spark.read.csv(i, header=True, inferSchema=True)
                test_df = assemble_dataframe(test, dataset.columns)
                prediction = tree_model.transform(test_df)
                prediction.show(5)
                evaluator = MulticlassClassificationEvaluator(
                    labelCol="label", predictionCol="prediction", metricName="accuracy")
                accuracy = evaluator.evaluate(prediction)
                print("Test Error = %g" % (1.0 - accuracy))
                break


cleanedDStream = words.foreachRDD(mapRdd)
ssc.start()
ssc.awaitTermination()
