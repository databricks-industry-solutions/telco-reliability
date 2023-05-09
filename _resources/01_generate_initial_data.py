# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %pip install dbldatagen

# COMMAND ----------

# MAGIC %run ./00-setup $reset_all_data=$reset_all_data $db_prefix=telco

# COMMAND ----------

reset_all = dbutils.widgets.get("reset_all_data") == "true"

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
globalIds = [row[0] for row in df_towers.select(df_towers.properties["GlobalID"]).collect()]


# COMMAND ----------

from datetime import datetime
import pyspark.sql.functions as F
import dbldatagen.distributions as dist
#CDR Historical Data Generation

if (not spark._jsparkSession.catalog().tableExists(f'{db_name}.{CDR_table_hist}')) or reset_all:
  #get subscriber ID to phone number mapping
  phone_numbers_df = spark.sql("select * from {}.{}".format(db_name, phone_numbers_table))
  #generate data
  partitions_requested = 36
  data_rows = 1000000000

  now = datetime.now()
  now_str = now.strftime("%Y-%m-%d %H:%M:%S")

  df_spec_cdr = (dg.DataGenerator(spark, name="CDR_records", rows=data_rows, partitions=partitions_requested)
                              .withIdOutput()
                              .withColumn("towerId", StringType(), values=globalIds, random=True)
                              .withColumn("subscriberId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                              .withColumn("otherId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                              .withColumn("type", StringType(), values=["call", "text"], random=True, weights=[2,8])
                              .withColumn("status_hidden", StringType(), values=["answered", "missed", "dropped"], random=True, weights=[8,3,1])
                              .withColumn("duration_hidden", IntegerType(), minValue=1, maxValue=110, random=True, distribution=dist.Gamma(1, 1))
                              .withColumn("event_ts", "timestamp", begin="2022-06-01 00:00:00", end="2022-12-31 00:00:00", interval="seconds=1", random=True)
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

  df_withText.write.format("delta").mode("overwrite").saveAsTable("{}.{}".format(db_name, CDR_table_hist))

# COMMAND ----------

#create Bronze, Silver, Gold of CDR Data
#bronze
if (not spark._jsparkSession.catalog().tableExists(f'{db_name}.{CDR_table_hist}_bronze')) or reset_all:
  df_cdr_raw = spark.read.table(f'{db_name}.{CDR_table_hist}')
  df_cdr_raw.write.format("delta").mode("overwrite").saveAsTable(f'{db_name}.{CDR_table_hist}_bronze')
  

#silver
if (not spark._jsparkSession.catalog().tableExists(f'{db_name}.{CDR_table_hist}_silver')) or reset_all:
  df_cdr_bronze = spark.read.table(f'{db_name}.{CDR_table_hist}_bronze')
  
  df_towers = spark.sql("select * from {0}.{1}".format(db_name, cell_tower_table))
  df_towers_select = df_towers.select(df_towers.properties.GlobalID.alias("GlobalID"), 
                                      df_towers.properties.LocCity.alias("City"), 
                                      df_towers.properties.LocCounty.alias("County"), df_towers.properties.LocState.alias("State"), 
                                      df_towers.geometry.coordinates[0].alias("Longitude"), 
                                      df_towers.geometry.coordinates[1].alias("Latitude"))
  
  df_cdr_towers_join = df_cdr_bronze.join(df_towers_select, df_cdr_bronze.towerId == df_towers_select.GlobalID)
                             
  df_cdr_towers_join.write.format("delta").mode("overwrite").saveAsTable(f'{db_name}.{CDR_table_hist}_silver')
  

#gold
if (not spark._jsparkSession.catalog().tableExists(f'{db_name}.{CDR_table_hist}_gold')) or reset_all:
  df_cdr_silver = spark.read.table(f'{db_name}.{CDR_table_hist}_silver')
  
  df_cdr_pivot_on_status_grouped_tower = (df_cdr_silver 
                                                 .groupBy(F.window("event_ts", "1 minute"), "towerId")                       
                                                 .agg(F.count(F.when(F.col("status") == "dropped", True)).alias("dropped"),   
                                                    F.count(F.when(F.col("status") == "answered", True)).alias("answered"),     
                                                    F.count(F.when(F.col("status") == "missed", True)).alias("missed"),         
                                                    F.count(F.when(F.col("type") == "text", True)).alias("text"),              
                                                    F.count(F.when(F.col("type") == "call", True)).alias("call"),              
                                                    F.count(F.lit(1)).alias("totalRecords_CDR"),                               
                                                    F.first("window.start").alias("window_start"),                              
                                                    F.first("Longitude").alias("Longitude"),                                    
                                                    F.first("Latitude").alias("Latitude"),                                      
                                                    F.first("City").alias("City"),                                              
                                                    F.first("County").alias("County"),                                          
                                                    F.first("State").alias("state")
                                                   )                                            
                                                  .withColumn("date", F.col("window_start")))
  
  df_cdr_pivot_on_status_grouped_tower.write.format("delta").mode("overwrite").saveAsTable(f'{db_name}.{CDR_table_hist}_gold')

# COMMAND ----------

#PCMD Sample Data
if (not spark._jsparkSession.catalog().tableExists(f'{db_name}.{PCMD_table_hist}')) or reset_all:
  def foreach_batch_function_pcmd(df, epoch_id):
    df.coalesce(1)
    df.write.mode("append").json(PCMD_dir)

  partitions_requested = 36
  data_rows = 1000000000

  now = datetime.now()
  now_str = now.strftime("%Y-%m-%d %H:%M:%S")

  if (not spark._jsparkSession.catalog().tableExists(f'{db_name}.{PCMD_table_hist}')) or reset_all:
    df_spec_pcmd = (dg.DataGenerator(spark, name="pcmdSample", partitions=partitions_requested, verbose=True)
                               .withIdOutput()
                               .withColumn("towerId", StringType(), values=globalIds, random=True)
                               .withColumn("subscriberId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                               .withColumn("ProcedureId", IntegerType(), values=[11, 15, 16], weights=[51, 5, 50])
                               .withColumn("ProcedureDuration", FloatType(), min=.0001, max=1)
                               .withColumn("event_ts", "timestamp", begin="2021-01-01 00:00:00", end=now_str, interval="seconds=1", random=True)
              )

    df_TestData_pcmd = df_spec_pcmd.build()

    df_TestData_pcmd.write.format("delta").mode("overwrite").saveAsTable("{}.{}".format(db_name, PCMD_table_hist))

# COMMAND ----------

#create Bronze, Silver, Gold of PCMD Data
#bronze
if (not spark._jsparkSession.catalog().tableExists(f'{db_name}.{PCMD_table_hist}_bronze')) or reset_all:
  df_pcmd_raw = spark.read.table(f'{db_name}.{PCMD_table_hist}')
  df_pcmd_raw.write.format("delta").mode("overwrite").saveAsTable(f'{db_name}.{PCMD_table_hist}_bronze')

  #silver
if (not spark._jsparkSession.catalog().tableExists(f'{db_name}.{PCMD_table_hist}_silver')) or reset_all:
  df_pcmd_bronze = spark.read.table(f'{db_name}.{PCMD_table_hist}_bronze')
  
  df_towers = spark.sql("select * from {0}.{1}".format(db_name, cell_tower_table))
  df_towers_select = df_towers.select(df_towers.properties.GlobalID.alias("GlobalID"), 
                                      df_towers.properties.LocCity.alias("City"), 
                                      df_towers.properties.LocCounty.alias("County"), df_towers.properties.LocState.alias("State"), 
                                      df_towers.geometry.coordinates[0].alias("Longitude"), 
                                      df_towers.geometry.coordinates[1].alias("Latitude"))
  
  
  df_pcmd_towers_join = df_pcmd_bronze.join(df_towers_select, df_pcmd_bronze.towerId == df_towers_select.GlobalID)
  
  df_pcmd_towers_join.write.format("delta").mode("overwrite").saveAsTable(f'{db_name}.{PCMD_table_hist}_silver')
  
#gold
if (not spark._jsparkSession.catalog().tableExists(f'{db_name}.{PCMD_table_hist}_gold')) or reset_all:
  df_pcmd_silver = spark.read.table(f'{db_name}.{PCMD_table_hist}_silver')
  
  df_pcmd_pivot_on_status_grouped_tower = (df_pcmd_silver 
                                            .groupBy(F.window("event_ts", "1 minute"), "towerId")                                                                     
                                            .agg(F.avg(F.when(F.col("ProcedureId") == "11", F.col("ProcedureDuration"))).alias("avg_dur_request_to_release_bearer"),  
                                                F.avg(F.when(F.col("ProcedureId") == "15", F.col("ProcedureDuration"))).alias("avg_dur_notification_data_sent_to_UE"),    
                                                F.avg(F.when(F.col("ProcedureId") == "16", F.col("ProcedureDuration"))).alias("avg_dur_request_to_setup_bearer"),         
                                                F.max(F.when(F.col("ProcedureId") == "11", F.col("ProcedureDuration"))).alias("max_dur_request_to_release_bearer"),       
                                                F.max(F.when(F.col("ProcedureId") == "15", F.col("ProcedureDuration"))).alias("max_dur_notification_data_sent_to_UE"),    
                                                F.max(F.when(F.col("ProcedureId") == "16", F.col("ProcedureDuration"))).alias("max_dur_request_to_setup_bearer"),         
                                                F.min(F.when(F.col("ProcedureId") == "11", F.col("ProcedureDuration"))).alias("min_dur_request_to_release_bearer"),       
                                                F.min(F.when(F.col("ProcedureId") == "15", F.col("ProcedureDuration"))).alias("min_dur_notification_data_sent_to_UE"),    
                                                F.min(F.when(F.col("ProcedureId") == "16", F.col("ProcedureDuration"))).alias("min_dur_request_to_setup_bearer"),         
                                                F.count(F.lit(1)).alias("totalRecords_PCMD"),                                                                             
                                                F.first("window.start").alias("window_start"),                                                                            
                                                F.first("Longitude").alias("Longitude"),                                                                                  
                                                F.first("Latitude").alias("Latitude"),                                                                                   
                                                F.first("City").alias("City"),                                                                                         
                                                F.first("County").alias("County"),             
                                                F.first("State").alias("state")
                                             )              
                                             .withColumn("date", F.col("window_start")))
  
  df_pcmd_pivot_on_status_grouped_tower.write.format("delta").mode("overwrite").saveAsTable(f'{db_name}.{PCMD_table_hist}_gold')

# COMMAND ----------

#CDR ML Data Generation Prep
cities = ["Denver", "Boulder"]
df_towers_den_boul = df_towers.filter(df_towers.properties.LocCity.isin(cities)).limit(6)

globalIds_filtered = [row[0] for row in df_towers_den_boul.select(df_towers_den_boul.properties["GlobalID"]).collect()]

#get subscriber ID to phone number mapping
phone_numbers_df = spark.sql("select * from {}.{}".format(db_name, phone_numbers_table))

# COMMAND ----------

#CDR ML Data Generation Prep
from datetime import date, timedelta
  
#generate data
partitions_requested = 36
data_rows = 1000000000

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

for dt in daterange(date(2021, 1, 1), date(2021, 12, 31)):
  if dt.weekday() in [0, 1, 2, 3, 4]:
    weekdays.append(dt.strftime("%Y-%m-%d"))
  else:
    weekends.append(dt.strftime("%Y-%m-%d"))
    

all_days = weekends + weekdays

# COMMAND ----------

#create baseline data
from pyspark.sql.types import LongType, IntegerType, StringType, FloatType, DateType
import dbldatagen as dg
import dbldatagen.distributions as dist

reset_ml_data = ((not spark._jsparkSession.catalog().tableExists(f'{db_name}.{CDR_table_ml}')) or reset_all)

if reset_ml_data:
  #generate data
  partitions_requested = 20
  data_rows = round(2920000000/2)


  #generate baseline data
  df_spec_cdr = (dg.DataGenerator(spark, name="baseline_CDR", rows=data_rows, partitions=partitions_requested)
                              .withIdOutput()
                              .withColumn("towerId", StringType(), values=globalIds_filtered, random=True)
                              .withColumn("subscriberId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                              .withColumn("otherId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                              .withColumn("type", StringType(), values=["call", "text"], random=True, weights=[2,8])
                              .withColumn("status_hidden", StringType(), values=["answered", "missed", "dropped"], random=True, weights=[8,3,1])
                              .withColumn("duration_hidden", IntegerType(), minValue=1, maxValue=110, random=True, distribution=dist.Gamma(1, 1))
                              .withColumn("hours_hidden", IntegerType(), values=hours, random=True, omit=True)
                              .withColumn("minutes_hidden", IntegerType(), minValue=0, maxValue=59, random=True, omit=True)
                              .withColumn("seconds_hidden", IntegerType(), minValue=0, maxValue=59, random=True, omit=True)
                              .withColumn("fulltime_hidden", StringType(), expr="concat(hours_hidden, ':', minutes_hidden, ':', seconds_hidden)", baseColumn=["hours_hidden", "minutes_hidden", "seconds_hidden"], omit=True)
                              .withColumn("date_hidden", StringType(), values=all_days, random=True, omit=True)
                              .withColumn("event_ts", TimestampType(), expr="to_timestamp(concat(date_hidden, ' ', fulltime_hidden))", baseColumn=["date_hidden", "fulltime_hidden"])
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

  df_withText.write.format("delta").mode("overwrite").saveAsTable("{}.{}".format(db_name, CDR_table_ml))

# COMMAND ----------

#create create weekend data
from pyspark.sql.types import LongType, IntegerType, StringType, FloatType, DateType, TimestampType
import dbldatagen as dg
import dbldatagen.distributions as dist

if reset_ml_data:
  #generate data
  partitions_requested = 20
  data_rows = round(2920000000/7)

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

  busier_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
  less_busy_hours = [1, 2, 3, 4, 5, 6, 7, 8, 22, 23, 24]
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

  df_withText.write.format("delta").mode("append").saveAsTable("{}.{}".format(db_name, CDR_table_ml))

# COMMAND ----------

#create create weekday data
from pyspark.sql.types import LongType, IntegerType, StringType, FloatType, DateType, TimestampType
import dbldatagen as dg
import dbldatagen.distributions as dist
  
if reset_ml_data:
  #generate data
  partitions_requested = 20
  data_rows = round(2920000000/2)

  #generate baseline data
  df_spec_cdr = (dg.DataGenerator(spark, name="weekend_CDR", rows=data_rows, partitions=partitions_requested)
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

  busier_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
  less_busy_hours = [1, 2, 3, 4, 5, 6, 7, 8, 22, 23, 24]
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

  df_withText.write.format("delta").mode("append").saveAsTable("{}.{}".format(db_name, CDR_table_ml))

# COMMAND ----------

import pyspark.sql.functions as F

def createWindowTable(sourceTable, targetTable, windowTime, towersTable):
  source_df = spark.sql("select * from {}".format(sourceTable))
  
  grouped_df = source_df.groupBy(F.window("event_ts", windowTime), "towerId")\
                        .agg(F.count(F.when(F.col("status") == "dropped", True)).alias("dropped"),   \
                          F.count(F.when(F.col("status") == "answered", True)).alias("answered"), \
                          F.count(F.when(F.col("status") == "missed", True)).alias("missed"),     \
                          F.count(F.when(F.col("type") == "text", True)).alias("text"),           \
                          F.count(F.when(F.col("type") == "call", True)).alias("call"),           \
                          F.count(F.lit(1)).alias("totalRecords_CDR"),                            \
                          F.first("window.start").alias("window_start"))                         \
                          .withColumn("datetime", (F.col("window_start")))
  
  df_towers = spark.sql("select * from {}".format(towersTable))
  df_towers_trunc = df_towers.select(df_towers.properties.GlobalID.alias("GlobalId"), df_towers.properties.LocCity.alias("City"), df_towers.properties.LocCounty.alias("County"), df_towers.properties.LocState.alias("State"), df_towers.geometry.coordinates[0].alias("Longitude"), df_towers.geometry.coordinates[1].alias("Latitude"))
  
  grouped_df_with_tower = grouped_df.join(df_towers_trunc, grouped_df.towerId == df_towers_trunc.GlobalId).drop("GlobalId")
  
  grouped_df_with_tower_ordered = grouped_df_with_tower.select("datetime",     \
                                                               "towerId",  \
                                                               "answered", \
                                                               "dropped",  \
                                                               "missed",   \
                                                               "call",     \
                                                               "text",     \
                                                               "totalRecords_CDR", \
                                                               "Latitude", \
                                                               "Longitude",\
                                                               "City",     \
                                                               "County",   \
                                                               "State")
  
  grouped_df_with_tower_ordered.write.format("delta").mode("overwrite").saveAsTable("{}".format(targetTable))

# COMMAND ----------

if reset_ml_data or (not spark._jsparkSession.catalog().tableExists(f'{db_name}.{CDR_table_hour_ml}')):
  #create hourly table
  sourceTable = "{}.{}".format(db_name, CDR_table_ml)
  targetTable = "{}.{}".format(db_name, CDR_table_hour_ml)
  windowTime = "1 hour"
  towersTable = "{}.{}".format(db_name, cell_tower_table)
  createWindowTable(sourceTable, targetTable, windowTime, towersTable)

if reset_ml_data or (not spark._jsparkSession.catalog().tableExists(f'{db_name}.{CDR_table_day_ml}')):
  #day table
  sourceTable = "{}.{}".format(db_name, CDR_table_ml)
  targetTable = "{}.{}".format(db_name, CDR_table_day_ml)
  windowTime = "1 day"
  towersTable = "{}.{}".format(db_name, cell_tower_table)
  createWindowTable(sourceTable, targetTable, windowTime, towersTable)

# COMMAND ----------


