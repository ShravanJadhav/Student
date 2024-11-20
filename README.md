## Data Science Project

import dagshub
dagshub.init(repo_owner='ShravanJadhav', repo_name='Student_Analysis', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)