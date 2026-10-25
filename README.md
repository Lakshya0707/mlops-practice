End-to-End MLOps Pipeline: Iris Flower Classification

A reproducible machine learning pipeline for the Iris dataset, built to practice real MLOps workflows: data and model versioning with DVC, experiment tracking and model registry with MLflow on DagsHub, a Flask prediction app, Docker packaging, and GitHub Actions CI.

Status: work in progress. The pipeline, tracking, registry, web app, tests and Docker packaging are done. Cloud deployment and monitoring are being added next (see Roadmap).

Tech Stack

Python, scikit-learn, pandas, DVC, MLflow, DagsHub, Flask, Gunicorn, Docker, GitHub Actions, Git

Pipeline
data_ingestion -> data_preprocessing -> feature_engineering -> model_building -> model_evaluation -> model_registry
Stage	What it does
data_ingestion	Loads the Iris CSV, encodes the target, splits into train/test
data_preprocessing	Fits StandardScaler on train only, transforms both sets, saves scaler.pkl
feature_engineering	Adds interaction features (PetalInteraction, SepalInteraction)
model_building	Trains a Random Forest using hyperparameters from params.yaml
model_evaluation	Computes accuracy, precision, recall, F1; logs params, metrics and model to MLflow
model_registry	Registers the model in the MLflow registry and sets the staging alias

Every stage is defined in dvc.yaml with its dependencies, parameters and outputs, so DVC reruns only what changed.

Key Features
Reproducible pipeline: DVC stages with parameters in params.yaml
Experiment tracking: every run logs parameters, metrics and artifacts to MLflow on DagsHub
Model registry with aliases: staging after training, production after tests pass (ci_scripts/promote_model.py)
No train-serve mismatch: the Flask app applies the same saved scaler and engineered features as training
Automated tests: model loading, feature signature, performance threshold and Flask endpoints
CI with GitHub Actions: runs the pipeline, tests and model promotion on push
Containerized app: slim Docker image, credentials passed at runtime through environment variables
Project Structure
.
├── src/                  # pipeline stages and logger
├── flask_app/            # Flask app, HTML template, inference preprocessing
├── tests/                # model and Flask tests
├── ci_scripts/           # model promotion script used by CI
├── .github/workflows/    # GitHub Actions workflow
├── dvc.yaml              # pipeline definition
├── params.yaml           # pipeline parameters
├── Dockerfile
└── requirements.txt
Getting Started

1. Clone and set up

bash
git clone https://github.com/Lakshya0707/mlops-practice.git
cd mlops-practice
python -m venv myvenv
myvenv\Scripts\activate          # Windows
pip install -r requirements.txt

2. Set DagsHub credentials (token from DagsHub: Settings > Tokens)

bash
set MLFLOW_TRACKING_USERNAME=<dagshub_username>
set MLFLOW_TRACKING_PASSWORD=<dagshub_token>

3. Run the pipeline

bash
dvc repro

4. Run the tests

bash
python -m unittest tests.test_model
python -m unittest tests.test_flask

5. Start the web app

bash
python flask_app/app.py

Open http://127.0.0.1:5000, enter the four measurements and get the predicted species.

Docker
bash
docker build -t capstone-app:latest .
docker run -p 8888:5000 \
  -e MLFLOW_TRACKING_USERNAME=<dagshub_username> \
  -e MLFLOW_TRACKING_PASSWORD=<dagshub_token> \
  capstone-app:latest

Open http://localhost:8888.

Experimenting

Change values in params.yaml (for example max_depth) and run dvc repro. DVC detects the change, reruns the affected stages and logs a new run in MLflow. Compare runs with:

bash
dvc params diff
dvc metrics diff

Latest metrics are saved in reports/metrics.json.

Roadmap

This project is a continuation of my MLOps learning, and the next steps are planned:

Deployment

 S3 as the DVC remote for data and model versioning
 Push the Docker image to AWS ECR from the CI pipeline
 Deploy the app on AWS EKS (Kubernetes Deployment and LoadBalancer Service) through CI/CD
 Store DagsHub credentials as a Kubernetes Secret
 Tag images with the commit SHA instead of only latest, for easy rollbacks

Monitoring

 Expose Prometheus metrics from the Flask app (request count, latency, prediction counts)
 Prometheus server to scrape the app
 Grafana dashboards for live monitoring

Quality and reliability

 Input validation for the prediction form (value ranges)
 Data validation and drift checks on incoming data
 Linting and pre-commit hooks in CI
 Least-privilege IAM permissions for the CI user
Author

Lakshya — GitHub
