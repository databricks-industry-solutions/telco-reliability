# Databricks notebook source
cloud_storage_path = "s3a://db-gtm-industry-solutions/data/CME/telco"

# COMMAND ----------

#table location definitions
db_name = "SOLACC_telco_reliability"

#locations of data streams
CDR_dir = cloud_storage_path + "/CDR"
PCMD_dir = cloud_storage_path + "/PCMD"

# COMMAND ----------

#table definitions for data generation
cell_tower_table = "cell_tower_geojson"
area_codes_table = "area_codes" 
phone_numbers_table = "phone_numbers"

forecast_table_hourly = "CDR_hour_forecast"

# COMMAND ----------

# MAGIC %sql
# MAGIC create database if not exists SOLACC_telco_reliability;
# MAGIC use SOLACC_telco_reliability;

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *

# COMMAND ----------

import pandas as pd
from prophet import Prophet
import logging

logging.getLogger('py4j').setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled=True

