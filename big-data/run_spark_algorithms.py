import csv
import os
import numpy as np
import datetime
from data.dataset import CompanyDataset
from spark_algorithms.data_processing import DataProcessing
from spark_algorithms.algorithms import Algorithms
from spark_algorithms.utils import get_hdfs_datasets, extract_strategy
from pyspark.sql import SparkSession

CSV_HEADERS = [
    "Name",
    "Strategy",
    "true_positives",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "Val_Accuracy",
    "Test_Accuracy",
    "Precision",
    "Recall",
    "F Score",
    "Cohen's Kappa coefficient",
]
results_for_csv = [CSV_HEADERS]

spark = SparkSession.builder.master("local[8]").appName("project").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# for strategy in ["tibi_mean", "tibi_median"]:
#     print(strategy)

#     provider = CompanyDataset(0.27, 0.055, strategy=strategy)
#     dataset = DataProcessing(spark, source="csv", dataset_provider_object=provider)
#     dataset.read_dataset()

#     algorithms = Algorithms(dataset, strategy)
#     algorithms.run_all()
#     results_for_csv.extend(algorithms.csv_rows)

paths = get_hdfs_datasets(spark)
for path in paths:
    print(path)
for path in paths[6:7]:
    try:
        strategy = extract_strategy(path[0])
        print(strategy)
        dataset = DataProcessing(spark, source="hdfs", hdfs_path=path)
        dataset.read_dataset()

        algorithms = Algorithms(dataset, strategy)
        algorithms.run_all()
        results_for_csv.extend(algorithms.csv_rows)
    except Exception as e:
        print("%s HAS FAILED!!!" % strategy)


with open(
    "spark_results/results_test_%s.csv"
    % datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"),
    "w+",
    newline="",
) as f:
    writer = csv.writer(f)
    writer.writerows(results_for_csv)
