# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./00-global-setup $reset_all_data=$reset_all_data $db_prefix=telco

# COMMAND ----------

#table location definitions
db_name = "solacc_telco_reliability"

#locations of data streams
CDR_dir = cloud_storage_path + "/CDR"
PCMD_dir = cloud_storage_path + "/PCMD"

#CDR_dir = "/mnt/field-demos/telco/CDR"
#PCMD_dir = "/mnt/field-demos/telco/PCMD"

# COMMAND ----------

""#table definitions for data generation
cell_tower_table = "cell_tower_geojson"
area_codes_table = "area_codes" 
phone_numbers_table = "phone_numbers"

#historical tables
CDR_table_hist = "CDR_hist"
PCMD_table_hist = "PCMD_hist"

CDR_table_ml = "CDR_ml"
CDR_table_hour_ml = "CDR_hour_gold_ml"
CDR_table_day_ml = "CDR_day_gold_ml"

forecast_table_hourly = "CDR_hour_forecast"

# COMMAND ----------

# MAGIC %sql
# MAGIC create database if not exists solacc_telco;
# MAGIC use solacc_telco;

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *

# COMMAND ----------

import pandas as pd
from prophet import Prophet
import logging

logging.getLogger('py4j').setLevel(logging.ERROR)

