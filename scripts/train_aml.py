# Azure ML training script
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml.entities import Environment

# Use DefaultAzureCredential (which includes support for Managed Identity)
credential = DefaultAzureCredential()

# Get the ML Client
ml_client = MLClient.from_config(credential=credential)

# Define the compute
cpu_compute_target = "cpu-cluster"

try:
    # Try to get the compute target
    cpu_cluster = ml_client.compute.get(cpu_compute_target)
    print(f"Found existing compute target: {cpu_compute_target}")
except Exception:
    print(f"Creating a new compute target: {cpu_compute_target}")
    
    # Define the compute configuration
    cpu_cluster = ml_client.compute.create_or_update(
        name=cpu_compute_target,
        type="amlcompute",
        size="STANDARD_DS3_v2",
        min_instances=0,
        max_instances=4,
        idle_time_before_scale_down=1800
    )

    cpu_cluster.wait_for_completion(show_output=True)

# Define the environment
custom_env = Environment(
    name="stock-prediction-env",
    description="Environment for stock prediction",
    conda_file="conda.yml"  # You need to create this file
)

# Create the command
job = command(
    code="./src",  # Location of source code
    command="python models/stock_prediction_pipeline.py",
    environment=custom_env,
    compute=cpu_compute_target,
    experiment_name="stock-prediction-experiment"
)

# Submit the job
returned_job = ml_client.jobs.create_or_update(job)
print(f"Submitted job: {returned_job}")

# Wait for the job to complete
ml_client.jobs.stream(returned_job.name)

# Get metrics
metrics = ml_client.jobs.get_metrics(returned_job.name)
for key, value in metrics.items():
    print(f"{key}: {value}")

# Register the model (assuming XGBoost is our best model)
model = ml_client.models.create_or_update(
    model=Model(
        name="stock_prediction_model",
        path="outputs/xgboost_model",
        type="custom_model"
    )
)
print(f"Registered model: {model.name} (version {model.version})")
