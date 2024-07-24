from pyspark.sql import SparkSession
# from databricks.connect.session import SparkSession

def get_taxis():
  print("Reading taxi trips data")
  spark = SparkSession.builder.getOrCreate()
  return spark.read.table("samples.nyctaxi.trips")

def main():
  get_taxis().show(5)

if __name__ == '__main__':
  main()
