# Databricks notebook source
# MAGIC %md This notebook is available at https://github.com/databricks-industry-solutions/telco-reliability. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/telco-network-analytics.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Network Demand Planning - Using Forecasting To Predict Tower Load
# MAGIC In the previous data engineering focused notebook, Per Call Measurement Data and Call Detail Records were ingested, transformed and cleansed, and ultimately aggregated at different time intervals. In this notebook, machine learning will be leveraged to generate forecasts using the gold tables generated from the pervious steps. 
# MAGIC
# MAGIC To be more specific, call detail records which have been aggregated into a gold table on a daily basis will be used to predict each tower's future activity. A benefit for a telecommunications company in doing this would be to find and plan for where outages can be expected or where those outages might affect the most customers. Ultimately, having accurate predictions for the future by tower can increase quality of service for customers and consequently reduce churn. 
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/telco-reliability/main/images/telco_pipeline_ml.png" width="1000"/>
# MAGIC
# MAGIC With the power of concurrent processing, we can easily scale our exploritory analysis and forecasting to data being generated from all the towers through horizontal scaling. 
# MAGIC
# MAGIC In terms of the technical aspects of this notebook, the <a href="https://facebook.github.io/prophet/docs/quick_start.html">Prophet library</a> will be used to predict future activity by each cell tower.

# COMMAND ----------

# MAGIC %run ./_resources/00-setup

# COMMAND ----------

#querying a few towers to explore forecasting
towerIds_list = [('{B8F4712B-F69B-40B2-BB38-0D6974424723}',), 
            ('{FA3B325C-25E1-42A5-8409-150E46FF3DF1}',), 
            ('{8C6DB461-40EB-4811-9151-C4118189BB0E}',), 
            ('{C128458D-A8A7-4C3B-A22A-6685F4DABA17}',),
            ('{8C3D0E39-4470-4E0B-97FB-EA351DC4FD71}',),
            ('{2630449E-8710-4134-B1C6-327EAE1978AB}',)]

towerIds_cols = ["towerId"]
towerIds = spark.createDataFrame(data=towerIds_list, schema = towerIds_cols)

# COMMAND ----------

#get the daily forecast for each tower and plot
def fitForecastDaily(towerId_row):
  towerId = towerId_row["towerId"]
  
  #use a Prophet model with a 95% confidence interval, looking for weekly and daily seasonality
  model = Prophet(
    interval_width=.95,
    growth="linear",
    weekly_seasonality=True,
    yearly_seasonality=False
  )

  #querying from the daily gold table to get total amount of activity by tower and date
  df = spark.sql("select CAST(datetime as date) as ds, sum(totalRecords_CDR) as y from cdr_stream_day_gold where year(datetime) = 2023 and towerId = '{}' group by ds".format(towerId))
  #do the actual fit in Pandas
  pandas_df = df.toPandas()
  model.fit(pandas_df)
  
  return model

#plot a daily model
def plotModelDaily(model, periods=90, freq='d', include_history=True):  
  future_pd = model.make_future_dataframe(periods=periods, freq=freq, include_history=True)

  #get forecast
  forecast_pd = model.predict(future_pd)
  trends_fig = model.plot_components(forecast_pd)
  predict_fig = model.plot(forecast_pd, xlabel='date', ylabel="records")

  #adjust figure to display dates from last year + the 90 day forecast
  xlim = predict_fig.axes[0].get_xlim()
  new_xlim = ( xlim[1]-(180.0+10.0), xlim[1]-10.0)
  predict_fig.axes[0].set_xlim(new_xlim)
  
  return forecast_pd

# COMMAND ----------

#call the above function for each tower
modelsByTowerDaily = []

for index, row in towerIds.toPandas().iterrows():
  modelsByTowerDaily.append(fitForecastDaily(row))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Forecasting by Day
# MAGIC We can plot the results of a forecasting model using all of 2021 data for a tower. With Prophet we can easily find the drivers that affect the prediction the most. For example with the first tower we can find that most of the activity seems to happen on the weekend (Saturday and Sunday) while the second tower that is plotted has more activity on the weekday (Monday-Friday).
# MAGIC
# MAGIC ### Weekend Tower
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/telco-reliability/main/images/weekend_component.png"/>
# MAGIC
# MAGIC ### Weekday Tower
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/telco-reliability/main/images/weekday_component.png"/>

# COMMAND ----------

plotModelDaily(modelsByTowerDaily[3])

# COMMAND ----------

plotModelDaily(modelsByTowerDaily[4])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forecasting on an hourly basis
# MAGIC As we see that most of the forecasting signal occurs on a more granular level (by weekday) we can also attempt to forecast on an hourly basis instead. In the graph below we see that in a weeks time some towers have more activity on the weekends while others are more active on the weekday. Within each day there are varying amounts of activity by the time of day (in hours). 

# COMMAND ----------

#using hourly gold table to predict next two weeks activity
logging.getLogger('py4j').setLevel(logging.ERROR)
def fitForecastHourly(towerId_row):
  towerId = towerId_row["towerId"]
  
  model = Prophet(
    interval_width=.95,
    growth="linear",
    daily_seasonality=True,
    weekly_seasonality=True
  )
  
  #take a segment of two weeks with hour granularity
  df = spark.sql("select datetime as ds, totalRecords_CDR as y from cdr_stream_hour_gold where datetime >= '2023-05-01' and datetime < '2023-05-15' and towerId = '{}'".format(towerId))
  pandas_df = df.toPandas()

  model.fit(pandas_df)
  
  return model

def plotModelHourly(model, periods=168, freq='H', include_history=True): 
  future_pd = model.make_future_dataframe(periods=168, freq='H', include_history=True)

  #get forecast
  forecast_pd = model.predict(future_pd)
  trends_fig = model.plot_components(forecast_pd)
  predict_fig = model.plot(forecast_pd, xlabel='date', ylabel="records")

  return forecast_pd

# COMMAND ----------

#calling above function
modelsByTowerHourly = []

for index, row in towerIds.toPandas().iterrows():
  modelsByTowerHourly.append(fitForecastHourly(row))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Hourly Predicions
# MAGIC When we forecast by hour, we can see the characteristics by day of a week in each tower where some towers have more activity on weekends while others are more heavily utilized on weekdays. However, with the granularity being at the hour level we can also predict what time of the day there will be more users connecting to the tower. We find that most activity starts occuring after 10am and starts decreasing after 9pm. 
# MAGIC
# MAGIC ### Hourly Components
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/telco-reliability/main/images/hour_of_day_component.png"/>

# COMMAND ----------

plotModelHourly(modelsByTowerHourly[3])

# COMMAND ----------

plotModelHourly(modelsByTowerHourly[4])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forecasting on a regular schedule
# MAGIC After our exploration of forecasting on a few sample towers, we've decided to do an hourly forecasting model as new data for each tower comes in. With the applyInPandas function, we can apply a Pandas UDF so that the forecasting is distributed amongst all the towers that we need to fit the model on. This leverages Spark by parallelizing the process and ultimately allowing us to scale as wide as needed so that we can handle a very large amount of data by scaling horizontally with our cluster.
# MAGIC
# MAGIC In the next steps, we define a Pandas UDF which creates a forecast on the a 2-week segment of data (by the hour) and then we save that prediction to a Delta table for future reference (with a training_date column so we know when we created the forecast). 

# COMMAND ----------

#create Pandas UDF
def forecastTowerHourlyResults(history_pd):
  model = Prophet(
    interval_width=.95,
    growth="linear",
    daily_seasonality=True,
    weekly_seasonality=True
  )
  
  model.fit(history_pd)
  
  future_pd = model.make_future_dataframe(
    periods=168, 
    freq='H', 
    include_history=True
    )
  
  forecast_pd = model.predict(future_pd)
  
  f_pd = forecast_pd[ ['ds','yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')
  
  h_pd = history_pd[['ds','towerId', 'y']].set_index('ds')
  
  results_pd = f_pd.join(h_pd, how='left')
  results_pd.reset_index(level=0, inplace=True)
  
  results_pd['towerId'] = history_pd['towerId'].iloc[0]
  
  return results_pd[['ds', 'towerId', 'y', 'yhat', 'yhat_upper', 'yhat_lower']]

# COMMAND ----------

#apply our forecast function to each tower

#define schema output
from pyspark.sql.types import *
from pyspark.sql.functions import current_timestamp
 
result_schema =StructType([
  StructField('ds',TimestampType()),
  StructField('towerId',StringType()),
  StructField('y',FloatType()),
  StructField('yhat',FloatType()),
  StructField('yhat_upper',FloatType()),
  StructField('yhat_lower',FloatType())
  ])

sql_statement = '''
  SELECT
    towerId,
    datetime as ds,
    sum(totalRecords_CDR) as y
  FROM cdr_stream_hour_gold 
  WHERE datetime >= '2023-05-01' and datetime < '2023-05-15'
  GROUP BY towerId, ds
  ORDER BY towerId, ds
'''

tower_activity_hourly_history = (spark
                                   .sql(sql_statement)
                                   .repartition(sc.defaultParallelism, ["towerId"])
                                ).cache()

results = (
  tower_activity_hourly_history
    .groupBy('towerId')
      .applyInPandas(forecastTowerHourlyResults, schema=result_schema)
    .withColumn('training_date', current_timestamp() )
    )

#write results to table with a training_date column
results.write.format("delta").mode("append").saveAsTable("cdr_hour_forecast")

display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ##(Optional) Anomaly Detection
# MAGIC For the same of an example, we can also use our forecasting model to do basic anomaly detection. In the cell below, we can use the previously generated Delta table of a forecast for each tower to detect whether the most recent activity is out of the bounds of the forecasting confidence interval. Hence, we have a column named "anomaly" which determines whether the reading was out of the confidence interval (which currently is set at .95). 
# MAGIC
# MAGIC As an example, we can see on our previous forecasts where the actual lies outside of the prediction confidence interval.
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/telco-reliability/main/images/anomoly_graph.png"/>

# COMMAND ----------

#now for anomoly detection
#1) here we have a regular job every hour to calculate CDR metrics so that the last hour can be put through the anomoly detection function
currentTimeWindow = '2023-05-15T00:00:00.000+0000'
df_current = spark.sql("SELECT * FROM cdr_stream_hour_gold WHERE datetime = '{}'".format(currentTimeWindow))

def anomolyDetectUDF_Func(actual, forecast_min, forecast_max):
  if actual < forecast_min or actual > forecast_max:
    return 1
  else:
    return 0
  
def anomolyImportanceUDF_Func(actual, forecast_min, forecast_max):
  if actual < forecast_min:
    return forecast_min - actual
  elif actual > forecast_max:
    return actual - forecast_max
  else:
    return None
  
anomolyDetectUDF = F.udf(anomolyDetectUDF_Func)
anomolyImportanceUDF = F.udf(anomolyImportanceUDF_Func)

def isAnomoly(df_current):
  mostRecentTraining = spark.sql("SELECT training_date FROM cdr_hour_forecast GROUP BY training_date ORDER BY training_date DESC LIMIT 1")
  currentTrainingDatetime = mostRecentTraining.collect()[0]['training_date']
  currentTimeWindow = df_current.select(F.first("datetime")).collect()[0]["first(datetime)"]
  
  df_forecast = spark.sql("SELECT * FROM cdr_hour_forecast WHERE training_date = '{}' AND ds = '{}'".format(currentTrainingDatetime, currentTimeWindow))
  
  #anomoly logic
  df_current_plus_forecast = df_current.join(df_forecast, ["towerId"])
  df_current_plus_forecast_with_anomoly = df_current_plus_forecast \
                                                     .withColumn("anomoly", anomolyDetectUDF(F.col("totalRecords_CDR"), F.col("cdr_hour_forecast.yhat_lower"), F.col("cdr_hour_forecast.yhat_upper"))) \
                                                     .withColumn("anomoly_importance", anomolyImportanceUDF(F.col("totalRecords_CDR"), F.col("cdr_hour_forecast.yhat_lower"), F.col("cdr_hour_forecast.yhat_upper")))
  
  return df_current_plus_forecast_with_anomoly
  
df_result = isAnomoly(df_current)
display(df_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC With the power of Delta and Spark, telecommunications companies can use Databricks to ingest large amounts of data on a regular basis through Delta Live Tables pipelines. From there, the data can be used to perform real time forecasting for future demand planning by tower or cell. Ultimately, this can help drive decisions for the business such as lowering maintenance response times and increasing the performance of the network through better deployment planning.
