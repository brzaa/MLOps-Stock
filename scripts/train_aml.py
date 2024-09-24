# Azure ML training script

import os
from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies

# Load the workspace from the saved config file
ws = Workspace.from_config(path="./config/config.json")

# Set up compute target
compute_name = "cpu-cluster"
if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    print("Found existing compute target.")
else:
    print("Creating a new compute target...")
    compute_config = AmlCompute.provisioning_configuration(
        vm_size='STANDARD_DS3_v2', 
        max_nodes=4,
        idle_seconds_before_scaledown=1800
    )
    compute_target = ComputeTarget.create(ws, compute_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

# Define the environment
environment = CondaDependencies()
environment.add_conda_package("python=3.8")
environment.add_conda_package("pip")
environment.add_pip_package("azureml-sdk")
environment.add_pip_package("pandas")
environment.add_pip_package("numpy")
environment.add_pip_package("scikit-learn")
environment.add_pip_package("xgboost")
environment.add_pip_package("lightgbm")
environment.add_pip_package("tensorflow")
environment.add_pip_package("mlflow")
environment.add_pip_package("yfinance")

# Create a run configuration
run_config = ScriptRunConfig(
    source_directory='./src',
    script='models/stock_prediction_pipeline.py',
    compute_target=compute_target,
    environment_definition=environment
)

# Set up the experiment
experiment = Experiment(workspace=ws, name='stock-prediction-experiment')

# Submit the experiment
run = experiment.submit(run_config)

# Display the run details
print(run.get_portal_url())

# Wait for the run to complete
run.wait_for_completion(show_output=True)

# Get metrics
metrics = run.get_metrics()
for key, value in metrics.items():
    print(f"{key}: {value}")

# Register the model (assuming XGBoost is our best model)
model = run.register_model(model_name='stock_prediction_model', model_path='outputs/xgboost_model')
print(f"Registered model: {model.name} (version {model.version})")
