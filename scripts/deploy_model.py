from azureml.core import Workspace, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.authentication import ServicePrincipalAuthentication
import os

def main():
    try:
        # Authenticate using service principal
        svc_pr = ServicePrincipalAuthentication(
            tenant_id=os.environ['AZURE_TENANT_ID'],
            service_principal_id=os.environ['AZURE_CLIENT_ID'],
            service_principal_password=os.environ['AZURE_CLIENT_SECRET'])

        # Load the Azure ML workspace
        ws = Workspace.get(
            name=os.environ['AML_WORKSPACE_NAME'],
            subscription_id=os.environ['AZURE_SUBSCRIPTION_ID'],
            resource_group=os.environ['AML_RESOURCE_GROUP'],
            auth=svc_pr
        )
        
        # Check if model exists
        model_name = "stock_prediction_model"
        if model_name not in ws.models:
            raise ValueError(f"Model '{model_name}' not found in workspace")
        
        # Load the registered model
        model = Model(ws, name=model_name)
        
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
            deployment_config=deployment_config,
            overwrite=True
        )
        
        service.wait_for_deployment(show_output=True)
        print(f"Deployment succeeded. Service URL: {service.scoring_uri}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
