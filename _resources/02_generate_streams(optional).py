# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %pip install dbldatagen

# COMMAND ----------

reset_all = dbutils.widgets.get("reset_all_data") == "true"

# COMMAND ----------

# MAGIC %run ./00-setup $reset_all_data=$reset_all_data $db_prefix=telco

# COMMAND ----------

#get tower IDs for data generation
import pyspark.sql.functions as F

df_towers = spark.sql("select * from {0}.{1}".format(db_name, cell_tower_table))

#get tower IDs in list format
globalIds = df_towers.select(df_towers.properties["GlobalID"]).rdd.flatMap(lambda x: x).collect()

#get subscriber ID to phone number mapping
phone_numbers_df = spark.sql("select * from {}.{}".format(db_name, phone_numbers_table))


# COMMAND ----------

# #RSSI Data Stream Generation

# from pyspark.sql.types import LongType, IntegerType, StringType, FloatType, DateType
# from datetime import datetime, timedelta
# import dbldatagen as dg
# import dbldatagen.distributions as dist

# #use for each batch function to write out streaming json
# def foreach_batch_function(df, epoch_id):
#   df.coalesce(1)
#   df.write.mode("append").json(RSSI_dir)

# #create data
# partitions_requested = 20
# data_rows = 10000

# now = datetime.now()
# now_str = now.strftime("%Y-%m-%d %H:%M:%S")

# df_spec_rssi = (dg.DataGenerator(spark, name="signalStrengthRecords", rows=data_rows, partitions=partitions_requested)
#                             .withIdOutput()
#                             .withColumn("towerId", StringType(), values=globalIds, random=True)
#                             .withColumn("RSRP", FloatType(), minValue=-30, maxValue=-120, random=True, distribution=dist.Gamma(5.0,3.0))
#                             .withColumn("RSRQ", FloatType(), minValue=-1, maxValue=-20.0, random=True, distribution=dist.Gamma(4.0, 2.0))
#                             .withColumn("SINR", FloatType(), minValue=-5, maxValue=13, random=True, distribution=dist.Gamma(10.0, 1.0))
#                             #.withColumn("event_ts", "timestamp", begin=now_str, end="2022-12-31 23:59:00", interval=timedelta(seconds=1))
#                             .withColumn("event_ts", "timestamp", expr="CURRENT_TIMESTAMP")
#                             )

# dfTestData_rssi = df_spec_rssi.build()

# #write out files
# dfTestData_rssi.writeStream.foreachBatch(foreach_batch_function).start()

# COMMAND ----------

CDR_dir

# COMMAND ----------

from pyspark.sql.types import LongType, IntegerType, StringType, FloatType, DateType
from datetime import datetime, timedelta
import dbldatagen as dg
import dbldatagen.distributions as dist

def path_exists(path):
  try:
    dbutils.fs.ls(path)
    return True
  except:
    return False

if reset_all and path_exists(CDR_dir):
  dbutils.fs.rm(CDR_dir, True)

if reset_all and path_exists(PCMD_dir):
  dbutils.fs.rm(PCMD_dir, True)

# COMMAND ----------

#CDR Data Stream Generation

#foreach batch function to write out JSON
# def foreach_batch_function_CDR(df, epoch_id):
#   df_withText = df.withColumn("status", F.when(F.col("type") == "text", None).otherwise(F.col("status_hidden"))) \
#     .drop("status_hidden")                                                                                       \
#     .withColumn("duration", F.when(F.col("type") == "text", None).otherwise(F.col("duration_hidden")))           \
#     .drop("duration_hidden")
  
#   df_withText.coalesce(1)
#   df_withText.write.mode("append").json(CDR_dir)
#   print("written")

if reset_all or (not path_exists(CDR_dir)):
  #generate data
  partitions_requested = 20
  data_rows = 1000000

  now = datetime.now()
  now_str = now.strftime("%Y-%m-%d %H:%M:%S")

  df_spec_cdr = (dg.DataGenerator(spark, name="signalStrengthRecords", partitions=partitions_requested, rows=data_rows, verbose=True)
                              .withIdOutput()
                              .withColumn("towerId", StringType(), values=globalIds, random=True)
                              .withColumn("subscriberId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                              .withColumn("otherId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                              .withColumn("type", StringType(), values=["call", "text"], random=True, weights=[2,8])
                              .withColumn("status", StringType(), values=["answered", "missed", "dropped"], random=True, weights=[8,3,1])
                              .withColumn("duration", IntegerType(), minValue=1, maxValue=110, random=True, distribution=dist.Gamma(1, 1))
                              #.withColumn("event_ts", "timestamp", begin=now_str, end="2022-12-31 23:59:00", interval=timedelta(seconds=1))
                              .withColumn("event_ts", "timestamp", expr="CURRENT_TIMESTAMP")
                              )

  df_TestData_cdr = df_spec_cdr.build()

  #extra stransformations to add phone numbers from IDs

  df_TestData_renamedId = df_TestData_cdr.withColumn("rId", F.col("Id")).drop("Id")

  df_phoneJoinedData = df_TestData_renamedId \
        .join(phone_numbers_df.select(F.col("phone_number").alias("subscriber_phone"), F.col("id").alias("id_sub")), F.col("subscriberId") == F.col("id_sub")) \
        .join(phone_numbers_df.select(F.col("phone_number").alias("other_phone"), F.col("id").alias("id_other")), F.col("otherId") == F.col("id_other"))

  #write out CDR data
  df_phoneJoinedData.write.json(CDR_dir)

# COMMAND ----------

#PCMD Sample Data
# def foreach_batch_function_pcmd(df, epoch_id):
#   df.coalesce(1)
#   df.write.mode("append").json(PCMD_dir)

if reset_all or (not path_exists(PCMD_dir)):
  now = datetime.now()
  now_str = now.strftime("%Y-%m-%d %H:%M:%S")

  partitions_requested = 20
  data_rows = 1000000

  df_spec_pcmd = (dg.DataGenerator(spark, name="pcmdSample", partitions=partitions_requested, rows=data_rows, verbose=True)
                             .withIdOutput()
                             .withColumn("towerId", StringType(), values=globalIds, random=True)
                             .withColumn("subscriberId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                             .withColumn("ProcedureId", IntegerType(), values=[11, 15, 16], weights=[51, 5, 50])
                             .withColumn("ProcedureDuration", FloatType(), min=.0001, max=1)
                             #.withColumn("event_ts", "timestamp", begin=now_str, end="2022-12-31 23:59:00", interval=timedelta(seconds=1))
                             .withColumn("event_ts", "timestamp", expr="CURRENT_TIMESTAMP")
            )

  df_TestData_pcmd = df_spec_pcmd.build()

  #write out PCMD data
  df_TestData_pcmd.write.json(PCMD_dir)

# COMMAND ----------


