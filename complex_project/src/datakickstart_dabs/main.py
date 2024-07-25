from pyspark.sql import SparkSession

def get_taxis():
  print("Reading taxi trips data")
  spark = SparkSession.builder.getOrCreate()
  return spark.read.table("samples.nyctaxi.trips")

def main():
  get_taxis().show(10)

if __name__ == '__main__':
  main()
