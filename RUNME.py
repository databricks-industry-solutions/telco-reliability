# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to illustrate the order of execution. Happy exploring! 
# MAGIC 🎉
# MAGIC
# MAGIC **Steps**
# MAGIC 1. Simply attach this notebook to a cluster and hit Run-All for this notebook. A multi-step job and the clusters used in the job will be created for you and hyperlinks are printed on the last block of the notebook. 
# MAGIC
# MAGIC 2. Run the accelerator notebooks: Feel free to explore the multi-step job page and **run the Workflow**, or **run the notebooks interactively** with the cluster to see how this solution accelerator executes. 
# MAGIC
# MAGIC     2a. **Run the Workflow**: Navigate to the Workflow link and hit the `Run Now` 💥. 
# MAGIC   
# MAGIC     2b. **Run the notebooks interactively**: Attach the notebook with the cluster(s) created and execute as described in the `job_json['tasks']` below.
# MAGIC
# MAGIC **Prerequisites** 
# MAGIC 1. You need to have cluster creation permissions in this workspace.
# MAGIC
# MAGIC 2. In case the environment has cluster-policies that interfere with automated deployment, you may need to manually create the cluster in accordance with the workspace cluster policy. The `job_json` definition below still provides valuable information about the configuration these series of notebooks should run with. 
# MAGIC
# MAGIC **Notes**
# MAGIC 1. The pipelines, workflows and clusters created in this script are not user-specific. Keep in mind that rerunning this script again after modification resets them for other users too.
# MAGIC
# MAGIC 2. If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators may require the user to set up additional cloud infra or secrets to manage credentials. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

pipeline_json_sql = {
         "clusters": [
              {
                  "label": "default",
                  "autoscale": {
                      "min_workers": 1,
                      "max_workers": 3
                  }
              }
          ],
          "development": True,
          "continuous": False,
          "edition": "advanced",
          "libraries": [
              {
                  "notebook": {
                      "path": f"01_Data Ingestion and Pipeline (SQL Version)",
                      "full_refresh": True
                  }
              }
          ],
          "name": "SOLACC-Telco Reliability-SQL",
          "storage": f"/databricks_solacc/telco_reliability/dlt_sql",
          "target": f"SOLACC_telco_reliability_sql",
          "allow_duplicate_names": "true"
      }

# COMMAND ----------

pipeline_id_sql = NotebookSolutionCompanion().deploy_pipeline(pipeline_json_sql, "", spark)

# COMMAND ----------

pipeline_json = {
          "clusters": [
              {
                  "label": "default",
                  "autoscale": {
                      "min_workers": 1,
                      "max_workers": 3
                  }
              }
          ],
          "development": True,
          "continuous": False,
          "edition": "advanced",
          "libraries": [
              {
                  "notebook": {
                      "path": f"01_Data Ingestion and Pipeline",
                      "full_refresh": True
                  }
              }
          ],
          "name": "SOLACC-Telco Reliability",
          "storage": f"/databricks_solacc/telco_reliability/dlt",
          "target": f"SOLACC_telco_reliability",
          "allow_duplicate_names": "true"
      }

# COMMAND ----------

pipeline_id = NotebookSolutionCompanion().deploy_pipeline(pipeline_json, "", spark)

# COMMAND ----------

job_json = {
        "timeout_seconds": 28800,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "CME"
        },
        "tasks": [
            {
                "pipeline_task": {
                    "pipeline_id": pipeline_id_sql
                },
                "task_key": "telco_reliability_01_sql"
            },
            {
                "pipeline_task": {
                    "pipeline_id": pipeline_id
                },
                "task_key": "telco_reliability_01"
            }
            ,
            {
                "job_cluster_key": "telco_reliability_cluster",
                "notebook_task": {
                    "notebook_path": f"02_Forecasting_with_Prophet"
                },
                "task_key": "telco_reliability_02",
                "depends_on": [
                    {
                        "task_key": "telco_reliability_01"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "telco_reliability_cluster",
                "new_cluster": {
                    "spark_version": "12.2.x-cpu-ml-scala2.12",
                "spark_conf": {
                    "spark.databricks.delta.formatCheck.enabled": "false"
                    },
                    "num_workers": 2,
                    "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_DS3_v2", "GCP": "n1-highmem-4"},
                    "custom_tags": {
                        "usage": "solacc_testing"
                    },
                }
            }
        ]
    }

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS databricks_solacc LOCATION '/databricks_solacc/'")
spark.sql(f"CREATE TABLE IF NOT EXISTS databricks_solacc.dbsql (path STRING, id STRING, solacc STRING)")
dbsql_config_table = "databricks_solacc.dbsql"

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
nsc = NotebookSolutionCompanion()
nsc.deploy_compute(job_json, run_job=run_job)
_ = nsc.deploy_dbsql("./resources/Telco CDR Overview with DLT Data.dbdash", dbsql_config_table, spark)

# COMMAND ----------


