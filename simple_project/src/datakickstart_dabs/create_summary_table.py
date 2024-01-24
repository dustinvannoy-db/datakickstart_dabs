def save_summary(df):
  df.selectExpr("cast(tpep_pickup_datetime as date) pickup_date", 
                "pickup_zip", "sum(trip_distance) total_distance", 
                "sum(fare_amount) total_amount"
              ).groupBy("pickup_date", "pickup_zip")
  df.mode("overwrite").saveAsTable("main.dustinvannoy_dev.trip_summary")

