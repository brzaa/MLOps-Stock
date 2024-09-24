# Model deployment script
from azureml.core import Workspace, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
def main():
    # Load the Azure ML workspace
    ws = Workspace.from_config()
    # Load the registered model
    model = Model(ws, name="stock_prediction_model")
    # Define an inference configuration
    inference_config = InferenceConfig(
        entry_script="scripts/score.py",
        environment=model.get_environment()
    )
    # Define the deployment configuration
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=1,
        memory_gb=1,
        auth_enabled=True
    )
    # Deploy the web service
    service = Model.deploy(
        workspace=ws,
        name="stock-prediction-service",
        models=[model],
        inference_config=inference_config,
        deployment_config=deployment_config
    )
    service.wait_for_deployment(show_output=True)
    print(f"Deployment succeeded. Service URL: {service.scoring_uri}")
if __name__ == "__main__":
    main()
