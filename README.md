# User-Controlled Tabular ML Platform on AWS

A platform where the human user makes all key modeling decisions. The system provides infrastructure and executes the user's specific configuration via Amazon SageMaker. No Auto-ML -- the user's choice is final.

## Architecture Overview

```
User --> Streamlit UI (Fargate) --> FastAPI Backend --> SageMaker Pipeline
                                        |                     |
                                        v                     v
                                  S3 Bucket <--------- Training Job
                                 (Versioned)          (ml.m5.large, ephemeral)
                                                            |
                                                            v
                                                      Model Registry
                                                  (Model Package Group)
```

## How It Works

### Built-in Models Mode

1. User uploads a CSV dataset through the Streamlit interface.
2. The app parses column headers and presents selection controls.
3. User selects: target column, columns to drop, ML model, and task type (classification/regression).
4. User clicks "Execute Pipeline" to trigger a SageMaker Pipeline.
5. SageMaker runs the training script with the user's exact configuration.
6. EDA plots and model metrics are saved to S3 and displayed in the UI.
7. The trained model is registered in the SageMaker Model Registry.

### Custom Script Mode

1. User switches to "Custom Script" mode via the workflow selector.
2. User uploads a Python training script (.py), a CSV dataset, and an optional requirements.txt.
3. User clicks "Start Training Job" to launch a standalone SageMaker training job.
4. The script runs on an ephemeral ml.m5.large instance using the SKLearn 1.2 container (Python 3.9).
5. Output artifacts (model, metrics, plots) are saved to S3 and displayed in the UI.

Custom scripts should follow SageMaker conventions:
- Read training data from the `--train` arg or `SM_CHANNEL_TRAIN` env var
- Save the model to `--model_dir` or `SM_MODEL_DIR` (`/opt/ml/model`)
- Save other outputs to `--output_dir` or `SM_OUTPUT_DATA_DIR` (`/opt/ml/output/data`)

## Supported Models

| Model | Library | Classification | Regression |
|-------|---------|----------------|------------|
| Logistic Regression | scikit-learn | Yes | No |
| Linear Regression | scikit-learn | No | Yes |
| XGBoost | xgboost | Yes | Yes |
| LightGBM | lightgbm | Yes | Yes |

## Project Structure

```
terraform/          Terraform infrastructure as code
  main.tf           Provider configuration
  variables.tf      Input variables
  outputs.tf        Output values
  s3.tf             S3 bucket with versioning
  ecr.tf            ECR container registry
  iam.tf            IAM roles (SageMaker, ECS)
  sagemaker.tf      Model Package Group
  ecs.tf            ECS Fargate cluster, task definition, service
  cloudwatch.tf     Monitoring: log groups, alarms, dashboard

app/                Streamlit and FastAPI application
  streamlit_app.py  User interface
  main.py           API backend
  requirements.txt  Python dependencies
  start.sh          Container entrypoint

pipeline/           SageMaker pipeline and training logic
  pipeline.py       Pipeline definition and orchestration
  universal_script.py  Model training and EDA script
  requirements.txt  Training dependencies

Dockerfile          Container image definition
.github/workflows/  CI/CD pipeline
README.md           This file
```

## Infrastructure

All infrastructure is provisioned with Terraform in the `eu-north-1` (Stockholm) region.

| Resource | Purpose | Cost |
|----------|---------|------|
| S3 Bucket | Data and artifact storage with versioning | Free-tier |
| ECR Repository | Docker image storage | Free-tier (500 MB) |
| ECS Fargate | Serverless container hosting | Pay-per-second (~$0.01/hr) |
| SageMaker ml.m5.large | Ephemeral training instances | ~$0.15-0.30/run |
| Model Package Group | Model versioning and registry | Free |
| CloudWatch | Logging, alarms, monitoring dashboard | Free-tier (5 GB logs, 10 alarms) |

For a demo session of a few hours, the total cost is under $1 for Fargate plus the SageMaker training costs.

### Scaling to Zero

When you are done using the platform, scale the service to zero to stop all Fargate charges:

```bash
aws ecs update-service \
  --cluster tabular-ml-platform-cluster \
  --service tabular-ml-platform-service \
  --desired-count 0 \
  --region eu-north-1
```

To start it again:

```bash
aws ecs update-service \
  --cluster tabular-ml-platform-cluster \
  --service tabular-ml-platform-service \
  --desired-count 1 \
  --region eu-north-1
```

After starting, find the public IP of the running task:

```bash
TASK_ARN=$(aws ecs list-tasks --cluster tabular-ml-platform-cluster --service tabular-ml-platform-service --region eu-north-1 --query 'taskArns[0]' --output text)
ENI_ID=$(aws ecs describe-tasks --cluster tabular-ml-platform-cluster --tasks $TASK_ARN --region eu-north-1 --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)
aws ec2 describe-network-interfaces --network-interface-ids $ENI_ID --region eu-north-1 --query 'NetworkInterfaces[0].Association.PublicIp' --output text
```

## Monitoring and Logging

All components are monitored via AWS CloudWatch. Terraform provisions the full monitoring stack.

### Log Groups

| Log Group | Source | Retention |
|-----------|--------|-----------|
| /ecs/tabular-ml-platform/app | Streamlit and FastAPI container logs | 14 days |
| /aws/sagemaker/TrainingJobs/tabular-ml-platform | SageMaker training job output | 14 days |
| /aws/sagemaker/Pipelines/tabular-ml-platform | SageMaker pipeline execution logs | 14 days |

### Alarms

| Alarm | Condition | Action |
|-------|-----------|--------|
| ECS CPU High | CPU utilization exceeds 80% | SNS notification |
| ECS Memory High | Memory utilization exceeds 80% | SNS notification |
| SageMaker Errors | Training job errors detected | SNS notification |

### Dashboard

A CloudWatch dashboard named `tabular-ml-platform-dashboard` provides a unified view of:
- ECS CPU and memory utilization
- Application error count
- Pipeline execution triggers
- Recent application logs
- Recent SageMaker training logs

Access the dashboard URL from Terraform outputs:
```bash
cd terraform && terraform output cloudwatch_dashboard_url
```

### Alert Notifications

To receive alarm notifications by email, set the `alert_email` variable:
```bash
terraform apply -var="alert_email=your@email.com"
```

## Prerequisites

- AWS account with free-tier eligibility
- AWS CLI configured with credentials
- Terraform >= 1.5.0
- Docker

## Setup

### 1. Infrastructure Provisioning

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

### 2. Initial Docker Image Build and Push

```bash
# Get the ECR repository URL from Terraform output
ECR_URL=$(cd terraform && terraform output -raw ecr_repository_url)

# Login to ECR
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin $ECR_URL

# Build and push
docker build -t tabular-ml-platform-app .
docker tag tabular-ml-platform-app:latest $ECR_URL:latest
docker push $ECR_URL:latest
```

### 3. Access the Application

After deployment, find the Fargate task public IP using the commands in the "Scaling to Zero" section above, then access:
```
http://<TASK_PUBLIC_IP>:8501
```

## CI/CD

The GitHub Actions workflow automatically:
1. Builds the Docker image on push to main branch
2. Pushes the image to Amazon ECR
3. Updates the ECS service to deploy the new image
4. Upserts the SageMaker Pipeline definition

Required GitHub Secrets:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION
- AWS_ACCOUNT_ID

## Cost Management

- ECS Fargate charges only when running (~$0.01/hr for 0.25 vCPU, 0.5 GB)
- Scale to zero when not in use to stop all Fargate costs
- SageMaker training instances are ephemeral (start, train, terminate)
- Each pipeline execution costs approximately $0.15-0.30
- S3 versioning provides data tracking without additional tools
- ECR lifecycle policy retains only the last 5 images

## Design Principles

- No Auto-ML: The user selects the exact model and parameters
- User explicitly chooses classification or regression task type
- AWS-native S3 versioning replaces DVC
- Serverless Fargate hosting with scale-to-zero capability
- Ephemeral SageMaker compute protects the budget
- Least-privilege IAM roles for all components
- All user choices are passed as SageMaker PipelineParameters
