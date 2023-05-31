# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %pip install dbldatagen

# COMMAND ----------

reset_all = dbutils.widgets.get("reset_all_data") == "true"

# COMMAND ----------

# MAGIC %run ./00-setup

# COMMAND ----------

#get area codes and create delta table
import requests
from delta.tables import DeltaTable
from pyspark.sql.functions import count

if (not spark._jsparkSession.catalog().tableExists(f'{db_name}.{area_codes_table}')) or reset_all:
  DeltaTable.createIfNotExists(spark) \
    .tableName("{0}.{1}".format(db_name, area_codes_table)) \
    .addColumn("AreaCode", "INT") \
    .execute()

  current_area_code_table = spark.sql(f'select count(*) as count from {db_name}.{area_codes_table}')

  if current_area_code_table.collect()[0][0] == 0:
    url = "https://raw.githubusercontent.com/ravisorg/Area-Code-Geolocation-Database/master/us-area-code-geo.csv"
    r = requests.get(url, allow_redirects=True)
    open('/dbfs/tmp/us-area-code-geo.csv', 'wb').write(r.content)

    area_code_df = spark.read.csv("/tmp/us-area-code-geo.csv", header=False, inferSchema= True)

    area_code_df.withColumnRenamed("_c0", "AreaCode").select("AreaCode").write.mode("overwrite").saveAsTable(f'{db_name}.{area_codes_table}')

# COMMAND ----------

import requests

#cell phone tower data
if (not spark._jsparkSession.catalog().tableExists(f'{db_name}.{cell_tower_table}')) or reset_all:
  url = "https://github.com/tomaszb-db/telco_fcc_tower_database/raw/main/cell_towers_v3.json.gz"
  r = requests.get(url)
  open('/dbfs/tmp/Cellular_Towers.json.gz', 'wb').write(r.content)
  
  cell_towers_df = spark.read.json("/tmp/Cellular_Towers.json.gz")
  
  cell_towers_df.write.mode("overwrite").saveAsTable(f'{db_name}.{cell_tower_table}')

# COMMAND ----------

import random
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import dbldatagen as dg
import pyspark.sql.functions as F

if (not spark._jsparkSession.catalog().tableExists(f'{db_name}.{phone_numbers_table}')) or reset_all:
  #random numbers with real area codes UDF
  area_codes_table_df = spark.sql("select * from {0}.{1}".format(db_name, area_codes_table))
  area_codes = area_codes_table_df.select("AreaCode").rdd.flatMap(lambda x: x).collect()

  df_phoneNums = (dg.DataGenerator(sparkSession=spark, name="phone_numbers", rows=1000000, partitions=4, randomSeedMethod="hash_fieldname")
                  .withColumn("phoneLastDigits", template='ddd-dddd', omit=True)
                  .withColumn("areaCode",  StringType(), values=area_codes, random=True, omit=True)
                  .withColumn("phone_number", StringType(), expr="concat(areaCode, '-', phoneLastDigits)", baseColumn=["areaCode", "phoneLastDigits"])
                 )

  df_phoneData = df_phoneNums.build()

  df_phoneData_withId = df_phoneData.withColumn("id", F.monotonically_increasing_id())
  df_phoneData_withId.write.mode("overwrite").saveAsTable("{0}.{1}".format(db_name, phone_numbers_table))

# COMMAND ----------

#get tower IDs for data generation
import pyspark.sql.functions as F

df_towers = spark.sql("select * from {0}.{1}".format(db_name, cell_tower_table))

#get tower IDs in list format
globalIds = df_towers.select(df_towers.properties["GlobalID"]).rdd.flatMap(lambda x: x).collect()

#get subscriber ID to phone number mapping
phone_numbers_df = spark.sql("select * from {}.{}".format(db_name, phone_numbers_table))


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
  
cdr_path_exist = path_exists(CDR_dir)

if reset_all and cdr_path_exist:
  dbutils.fs.rm(CDR_dir, True)

if reset_all and path_exists(PCMD_dir):
  dbutils.fs.rm(PCMD_dir, True)

# COMMAND ----------

#CDR Data Stream Generation

if reset_all or (not cdr_path_exist):
  #generate data
  partitions_requested = 20
  data_rows = 100000000

  df_spec_cdr = (dg.DataGenerator(spark, name="CDR_data", partitions=partitions_requested, rows=data_rows, verbose=True)
                              .withIdOutput()
                              .withColumn("towerId", StringType(), values=globalIds, random=True)
                              .withColumn("subscriberId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                              .withColumn("otherId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                              .withColumn("type", StringType(), values=["call", "text"], random=True, weights=[2,8])
                              .withColumn("status_hidden", StringType(), values=["answered", "missed", "dropped"], random=True, weights=[21,10,1])
                              .withColumn("duration_hidden", IntegerType(), minValue=1, maxValue=110, random=True, distribution=dist.Gamma(1, 1))
                              .withColumn("event_ts", "timestamp", begin="2022-01-01 00:00:00", end="2022-12-31 00:00:00", interval=timedelta(seconds=43))
                              )

  df_TestData_cdr = df_spec_cdr.build()

  #extra stransformations to add phone numbers from IDs

  df_TestData_renamedId = df_TestData_cdr.withColumn("rId", F.col("Id")).drop("Id")

  df_phoneJoinedData = df_TestData_renamedId \
        .join(phone_numbers_df.select(F.col("phone_number").alias("subscriber_phone"), F.col("id").alias("id_sub")), F.col("subscriberId") == F.col("id_sub")) \
        .join(phone_numbers_df.select(F.col("phone_number").alias("other_phone"), F.col("id").alias("id_other")), F.col("otherId") == F.col("id_other"))

  df_withText = df_phoneJoinedData.withColumn("status", F.when(F.col("type") == "text", None).otherwise(F.col("status_hidden"))) \
      .drop("status_hidden")                                                                                       \
      .withColumn("duration", F.when(F.col("type") == "text", None).otherwise(F.col("duration_hidden")))           \
      .drop("duration_hidden")

  #write out CDR data
  df_withText.write.json(CDR_dir)

# COMMAND ----------

#filter towers for cities for ML piece
globalIds_filtered = ['{B8F4712B-F69B-40B2-BB38-0D6974424723}', 
            '{FA3B325C-25E1-42A5-8409-150E46FF3DF1}', 
            '{8C6DB461-40EB-4811-9151-C4118189BB0E}', 
            '{C128458D-A8A7-4C3B-A22A-6685F4DABA17}',
            '{8C3D0E39-4470-4E0B-97FB-EA351DC4FD71}',
            '{2630449E-8710-4134-B1C6-327EAE1978AB}']

# COMMAND ----------

#CDR ML Data Generation Prep
from datetime import date, timedelta

#hour distributions
hours = [x for x in range(0, 24)]
busier_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
less_busy_hours = [1, 2, 3, 4, 5, 6, 7, 8, 22, 23, 24]

#weekdays vs weekends
weekends = []
weekdays = []

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

for dt in daterange(date(2023, 1, 1), date(2023, 5, 31)):
  if dt.weekday() in [0, 1, 2, 3, 4]:
    weekdays.append(dt.strftime("%Y-%m-%d"))
  else:
    weekends.append(dt.strftime("%Y-%m-%d"))
    

all_days = weekends + weekdays

# COMMAND ----------

print(weekdays)

# COMMAND ----------

print(weekends)

# COMMAND ----------

print(weekends)

# COMMAND ----------

#create create weekend data
from pyspark.sql.types import LongType, IntegerType, StringType, FloatType, DateType, TimestampType
import dbldatagen as dg
import dbldatagen.distributions as dist


if reset_all or (not cdr_path_exist):
  #generate data
  partitions_requested = 20
  data_rows = round(292000000/7)

  #generate baseline data
  df_spec_cdr = (dg.DataGenerator(spark, name="weekend_CDR", rows=data_rows, partitions=partitions_requested)
                              .withIdOutput()
                              .withColumn("towerId", StringType(), values=globalIds_filtered, random=True, weights=[2, 2, 7, 2, 8, 2])
                              .withColumn("subscriberId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                              .withColumn("otherId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                              .withColumn("type", StringType(), values=["call", "text"], random=True, weights=[2,8])
                              .withColumn("status_hidden", StringType(), values=["answered", "missed", "dropped"], random=True, weights=[30,16,1])
                              .withColumn("duration_hidden", IntegerType(), minValue=1, maxValue=110, random=True, distribution=dist.Gamma(1, 1))
                              .withColumn("hours_hidden", IntegerType(), values=hours, random=True, weights=[1, 1, 1, 1, 1, 1, 2, 3, 3, 6, 9, 7, 9, 7, 7, 8, 7, 8, 5, 6, 7, 8, 7, 6], omit=True)
                              .withColumn("minutes_hidden", IntegerType(), minValue=0, maxValue=59, random=True, omit=True)
                              .withColumn("seconds_hidden", IntegerType(), minValue=0, maxValue=59, random=True, omit=True)
                              .withColumn("fulltime_hidden", StringType(), expr="concat(hours_hidden, ':', minutes_hidden, ':', seconds_hidden)", baseColumn=["hours_hidden", "minutes_hidden", "seconds_hidden"], omit=True)
                              .withColumn("date_hidden", StringType(), values=weekends, random=True, omit=True)
                              .withColumn("event_ts", TimestampType(), expr="to_timestamp(concat(date_hidden, ' ', fulltime_hidden))", baseColumn=["date_hidden", "fulltime_hidden"])
                              )

  #extra stransformations to add phone numbers from IDs
  df_TestData_cdr = df_spec_cdr.build()

  df_TestData_renamedId = df_TestData_cdr.withColumn("rId", F.col("Id")).drop("Id")

  df_phoneJoinedData = df_TestData_renamedId \
        .join(phone_numbers_df.select(F.col("phone_number").alias("subscriber_phone"), F.col("id").alias("id_sub")), F.col("subscriberId") == F.col("id_sub")) \
        .join(phone_numbers_df.select(F.col("phone_number").alias("other_phone"), F.col("id").alias("id_other")), F.col("otherId") == F.col("id_other"))

  df_withText = df_phoneJoinedData.withColumn("status", F.when(F.col("type") == "text", None).otherwise(F.col("status_hidden"))) \
      .drop("status_hidden")                                                                                       \
      .withColumn("duration", F.when(F.col("type") == "text", None).otherwise(F.col("duration_hidden")))           \
      .drop("duration_hidden")

  df_withText.write.mode("overwrite").json(CDR_dir+"/testdir/")

# COMMAND ----------

#create create weekday data
from pyspark.sql.types import LongType, IntegerType, StringType, FloatType, DateType, TimestampType
import dbldatagen as dg
import dbldatagen.distributions as dist

if reset_all or (not cdr_path_exist):
  #generate data
  partitions_requested = 20
  data_rows = round(292000000/7)


  df_spec_cdr2 = (dg.DataGenerator(spark, name="weekday_CDR", rows=data_rows, partitions=partitions_requested)
                              .withIdOutput()
                              .withColumn("towerId", StringType(), values=globalIds_filtered, random=True, weights=[10, 9, 3, 12, 3, 11])
                              .withColumn("subscriberId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                              .withColumn("otherId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                              .withColumn("type", StringType(), values=["call", "text"], random=True, weights=[2,8])
                              .withColumn("status_hidden", StringType(), values=["answered", "missed", "dropped"], random=True, weights=[30,16,1])
                              .withColumn("duration_hidden", IntegerType(), minValue=1, maxValue=110, random=True, distribution=dist.Gamma(1, 1))
                              .withColumn("hours_hidden", IntegerType(), values=hours, random=True, weights=[1, 1, 1, 1, 1, 1, 2, 3, 3, 6, 9, 7, 9, 7, 7, 8, 7, 8, 5, 6, 7, 8, 7, 6], omit=True)
                              .withColumn("minutes_hidden", IntegerType(), minValue=0, maxValue=59, random=True, omit=True)
                              .withColumn("seconds_hidden", IntegerType(), minValue=0, maxValue=59, random=True, omit=True)
                              .withColumn("fulltime_hidden", StringType(), expr="concat(hours_hidden, ':', minutes_hidden, ':', seconds_hidden)", baseColumn=["hours_hidden", "minutes_hidden", "seconds_hidden"], omit=True)
                              .withColumn("date_hidden", StringType(), values=weekdays, random=True, omit=True)
                              .withColumn("event_ts", TimestampType(), expr="to_timestamp(concat(date_hidden, ' ', fulltime_hidden))", baseColumn=["date_hidden", "fulltime_hidden"])
                              )

  df_TestData_cdr2 = df_spec_cdr2.build()

  #extra stransformations to add phone numbers from IDs

  df_TestData_renamedId = df_TestData_cdr2.withColumn("rId", F.col("Id")).drop("Id")

  df_phoneJoinedData = df_TestData_renamedId \
        .join(phone_numbers_df.select(F.col("phone_number").alias("subscriber_phone"), F.col("id").alias("id_sub")), F.col("subscriberId") == F.col("id_sub")) \
        .join(phone_numbers_df.select(F.col("phone_number").alias("other_phone"), F.col("id").alias("id_other")), F.col("otherId") == F.col("id_other"))

  df_withText = df_phoneJoinedData.withColumn("status", F.when(F.col("type") == "text", None).otherwise(F.col("status_hidden"))) \
      .drop("status_hidden")                                                                                       \
      .withColumn("duration", F.when(F.col("type") == "text", None).otherwise(F.col("duration_hidden")))           \
      .drop("duration_hidden")

  df_withText.write.mode("append").json(CDR_dir)

# COMMAND ----------

#PCMD Sample Data

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
                             .withColumn("event_ts", "timestamp", begin="2022-01-01 00:00:00", end=now_str, interval="seconds=1", random=True
            )
  )

  df_TestData_pcmd = df_spec_pcmd.build()

  #write out PCMD data
  df_TestData_pcmd.write.mode("overwrite").json(PCMD_dir)

# COMMAND ----------


