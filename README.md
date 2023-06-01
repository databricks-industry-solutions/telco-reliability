![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

## Telecommunications Reliability Metrics

**Telecommunications LTE Architecture**
<br>



The modern telecommunications network consists of the Base Station also known as the **eNodeB (Evolved Node B)** is the hardware that communicates directly with the **UE (User Enitity such as a Mobile Phone)**. The **MME (Mobility Management Entity)** manages the entire process from a cell phone making a connection to a network to a paging message being sent to the mobile phone.  
<img style="margin: auto" src="https://raw.githubusercontent.com/databricks-industry-solutions/telco-reliability/main/images/Telco_simple.png" width="1200"/>

**Use Case Overview**
* Telecommunications services collect many different forms of data to observe overall network reliability as well as to predict how best to expand the network to reach more customers. Some typical types of data collected are:
  - **PCMD (Per Call Measurement Data):** granular details of all network processes as MME (Mobility Management Entity) manages processes between the UE and the rest of the network
  - **CDR (Call Detail Records):** high level data describing call and SMS activity with fields such as phone number origin, phone number target, status of call/sms, duration, etc. 
* This data can be collected and used in provide a full view of the health of each cell tower in the network as well as the network as a whole. 
* **Note:** for this demo we will be primarily focused on CDR data but will also have a small sample of what PCMD could look like.

**Business Impact of Solution**
* **Ease of Scaling:** with large amounts of data being generated by a telecommunications system, Databricks can provide the ability to scale so that the data can be reliably ingested and analyzed.  
* **Greater Network Reliability:** with the ability to monitor and predict dropped communications and more generally network faults, telecommunications providers can ultimately deliver better service for their customers and reduce churn.

___
<tomasz.bacewicz@databricks.com>

___

**Full Architecture from Ingestion to Analytics and Machine Learning**
<img src="https://raw.githubusercontent.com/databricks-industry-solutions/telco-reliability/main/images/telco_pipeline_full.png" width="1000"/>

___

&copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| prophet                                 | Timeseries forecasting      | MIT        | https://github.com/facebook/prophet                |

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format. 
3. Execute the multi-step-job to see how the pipeline runs. 
4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
