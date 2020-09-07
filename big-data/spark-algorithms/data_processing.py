import numpy as np

from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.sql.functions import monotonically_increasing_id, row_number, rand


def convert_to_spark_df(spark, data, input_cols):
    dataset_spark = spark.sparkContext.parallelize(data)
    df = dataset_spark.map(lambda x: x.tolist()).toDF(input_cols + ["label"])
    df = df.withColumn("label", df["label"].cast(IntegerType()))
    return df


def assemble_dataframe(df):
    columns = df.columns
    columns.remove("label")
    assembler = VectorAssembler(inputCols=columns, outputCol="features")

    output = assembler.transform(df)
    output = output.select(["features", "label"])
    return output


def concatenate_features_with_label(x, y):
    x_with_index = x.withColumn(
        "row_index", row_number().over(Window.orderBy(monotonically_increasing_id()))
    )
    y_with_index = y.withColumn(
        "row_index", row_number().over(Window.orderBy(monotonically_increasing_id()))
    )
    concatenated = x_with_index.join(y_with_index, on=["row_index"]).drop("row_index")
    return concatenated


class DataProcessing:
    def __init__(
        self, spark, source="csv", hdfs_path=None, dataset_provider_object=None
    ):
        self.spark = spark
        self.source = source
        self.hdfs_path = hdfs_path
        self.dataset_provider_object = dataset_provider_object
        self.full_train = None
        self.full_valid = None
        self.full_test = None
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.columns = None
        self.train_assembled = None
        self.valid_assembled = None
        self.test_assembled = None

    def read_dataset(self):
        if self.source == "csv":
            # dataset = CompanyDataset(0.001, 0.45, strategy=self.strategy)
            dataset = self.dataset_provider_object

            # self.full_train = np.c_[dataset.train_input, dataset.train_output]
            # self.full_valid = np.c_[dataset.valid_input, dataset.valid_output]
            # self.full_test = np.c_[dataset.test_input, dataset.test_output]

            self.full_train = np.c_[dataset.train_input, dataset.train_output]
            self.full_valid = np.c_[dataset.valid_input, dataset.valid_output]
            self.full_test = np.c_[
                dataset.future_data, np.zeros(dataset.future_data.shape[0])
            ]

            self.train_df = convert_to_spark_df(
                self.spark, self.full_train, dataset.columns
            )
            self.valid_df = convert_to_spark_df(
                self.spark, self.full_valid, dataset.columns
            )
            self.test_df = convert_to_spark_df(
                self.spark, self.full_test, dataset.columns
            )

            self.train_assembled = assemble_dataframe(self.train_df)
            self.valid_assembled = assemble_dataframe(self.valid_df)
            self.test_assembled = assemble_dataframe(self.test_df)
            self.columns = dataset.columns
        elif self.source == "hdfs":
            self.train_df = self.spark.read.csv(
                self.hdfs_path[0], inferSchema=True, header=True
            )
            self.valid_df = self.spark.read.csv(
                self.hdfs_path[1], inferSchema=True, header=True
            )
            self.test_df = self.spark.read.csv(
                self.hdfs_path[2], inferSchema=True, header=True
            )

            schema = StructType([StructField("label", FloatType(), True)])

            labels_train = self.spark.read.csv(
                "hdfs://localhost:9000/user/florin/datasets/y_train.csv",
                header=True,
                schema=schema,
            )

            labels_valid = self.spark.read.csv(
                "hdfs://localhost:9000/user/florin/datasets/y_val.csv",
                header=True,
                schema=schema,
            )

            labels_test = self.spark.read.csv(
                "hdfs://localhost:9000/user/florin/datasets/y_test.csv",
                header=True,
                schema=schema,
            )

            self.full_train = concatenate_features_with_label(
                self.train_df, labels_train
            )
            self.full_valid = concatenate_features_with_label(
                self.valid_df, labels_valid
            )
            self.full_test = concatenate_features_with_label(self.test_df, labels_test)

            self.train_assembled = assemble_dataframe(self.full_train)
            self.train_assembled = self.train_assembled.orderBy(rand())

            self.valid_assembled = assemble_dataframe(self.full_valid)
            self.valid_assembled = self.valid_assembled.orderBy(rand())

            self.test_assembled = assemble_dataframe(self.full_test)

            self.columns = self.train_df.columns

    # return self.train_assembled, self.test_assembled
