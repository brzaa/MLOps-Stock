# Model deployment script
from azureml.core import Workspace, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

def main():
    # Load the Azure ML workspace
    ws = Workspace.from_config()

    # Load the registered model
    model = Model(ws, name="stock_prediction_model")

    # Define the environment
    env = Environment(name="stock-prediction-env")
    env.python.conda_dependencies = CondaDependencies()
    env.python.conda_dependencies.add_conda_package("python=3.9")
    env.python.conda_dependencies.add_pip_package("azureml-sdk[automl]==1.48.0")
    env.python.conda_dependencies.add_pip_package("pandas==1.5.3")
    env.python.conda_dependencies.add_pip_package("numpy==1.21.0")
    env.python.conda_dependencies.add_pip_package("scikit-learn==1.5.1")
    env.python.conda_dependencies.add_pip_package("xgboost==1.6.2")
    env.python.conda_dependencies.add_pip_package("mlflow==2.3.2")
    env.python.conda_dependencies.add_pip_package("joblib==1.2.0")
    env.python.conda_dependencies.add_pip_package("yfinance==0.2.12")
    # Add additional pip packages as needed

    # Define an inference configuration
    inference_config = InferenceConfig(
        entry_script="scripts/score.py",
        environment=env
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
