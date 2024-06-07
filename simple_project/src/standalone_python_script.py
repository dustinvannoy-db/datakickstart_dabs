import os

# from pyspark.sql import SparkSession
# from pyspark.dbutils import DBUtils

def get_spark(): #-> SparkSession:
  return spark

def get_taxis():
  print("Reading taxi trips data")
  return spark.read.table("samples.nyctaxi.trips")


def save_summary(df):
  df2 = df.select(
        df.tpep_pickup_datetime.cast("date").alias("pickup_date"), 
        df.pickup_zip, 
        df.trip_distance,
        df.fare_amount
        )
  df2.createOrReplaceTempView("trip_tmp")

  df_agg = spark.sql("""
          SELECT 
            pickup_date, 
            pickup_zip, 
            SUM(trip_distance) as trip_distance, 
            SUM(fare_amount) as fare_amount
          FROM trip_tmp
          GROUP BY pickup_date, pickup_zip
        """)
        
  df_agg.write.mode("overwrite").saveAsTable("main.dustinvannoy_dev.trip_summary")
  
  df_agg.show()

global spark
spark = get_spark()
print("Running on cluster: ", dbutils._cluster_id)

df = get_taxis()
save_summary(df)

#-----MISC COMMANDS -----#
is_databricks = True if os.getenv("DATABRICKS_RUNTIME_VERSION") is not None else False
print(is_databricks)

secret = dbutils.secrets.get(scope='fieldeng', key='dustin-secret')
print(len(secret))
