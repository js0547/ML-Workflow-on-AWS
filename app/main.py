"""
FastAPI Backend for the Tabular ML Platform.

Provides REST endpoints for:
- Uploading CSV data to S3
- Triggering SageMaker Pipeline executions
- Monitoring pipeline execution status
- Retrieving EDA plots and metrics from S3

All endpoints are consumed by the Streamlit frontend.
"""

import io
import json
import logging
import os
import time
from datetime import datetime

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration from environment variables (set by ECS task definition)
# ------------------------------------------------------------------

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-north-1")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")
MODEL_PACKAGE_GROUP = os.environ.get("MODEL_PACKAGE_GROUP", "tabular-ml-platform-models")
PIPELINE_NAME = os.environ.get("PIPELINE_NAME", "tabular-ml-platform-pipeline")

# ------------------------------------------------------------------
# AWS Clients
# ------------------------------------------------------------------

s3_client = boto3.client("s3", region_name=AWS_REGION)
sagemaker_client = boto3.client("sagemaker", region_name=AWS_REGION)

# ------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------

app = FastAPI(
    title="Tabular ML Platform API",
    description="Backend API for the User-Controlled Tabular ML Platform",
    version="1.0.0",
)

# Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Request/Response Models
# ------------------------------------------------------------------

class TriggerRequest(BaseModel):
    """Request body for triggering a pipeline execution."""
    s3_uri: str
    target_column: str
    dropped_columns: str
    model_type: str
    task_type: str


# Valid model/task combinations. Used for server-side validation.
_VALID_MODEL_TASKS = {
    "logistic_regression": ["classification"],
    "linear_regression": ["regression"],
    "xgboost": ["classification", "regression"],
    "lightgbm": ["classification", "regression"],
}


class TriggerResponse(BaseModel):
    """Response body after triggering a pipeline execution."""
    execution_arn: str
    status: str
    message: str


class StatusResponse(BaseModel):
    """Response body for pipeline execution status."""
    execution_arn: str
    status: str
    steps: list
    start_time: str = ""
    end_time: str = ""
    parameters: dict = {}


class ArtifactResponse(BaseModel):
    """Response body for pipeline artifacts."""
    correlation_heatmap_url: str = ""
    missing_value_matrix_url: str = ""
    data_summary: dict = {}
    evaluation_metrics: dict = {}
    model_package_arn: str = ""


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """Health check endpoint for ECS/load balancer."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "region": AWS_REGION,
        "bucket": S3_BUCKET,
        "pipeline": PIPELINE_NAME,
    }


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file to S3.
    Returns the S3 URI and parsed column headers for the UI selectors.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        # Read file content
        content = await file.read()

        # Parse headers using pandas
        df = pd.read_csv(io.BytesIO(content))
        columns = list(df.columns)
        row_count = len(df)

        # Generate a unique S3 key with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        s3_key = "data/{}_{}" .format(timestamp, file.filename)
        s3_uri = "s3://{}/{}".format(S3_BUCKET, s3_key)

        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=content,
            ContentType="text/csv",
        )

        logger.info("Uploaded %s to %s (%d rows, %d columns)",
                     file.filename, s3_uri, row_count, len(columns))

        return {
            "s3_uri": s3_uri,
            "filename": file.filename,
            "columns": columns,
            "row_count": row_count,
            "column_count": len(columns),
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The CSV file is empty.")
    except Exception as e:
        logger.error("Upload failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Upload failed: {}".format(str(e)))


@app.post("/api/trigger", response_model=TriggerResponse)
def trigger_pipeline(request: TriggerRequest):
    """
    Trigger a SageMaker Pipeline execution with the user's selections.
    All modeling decisions come from the user -- no auto-ML logic.
    """
    # Server-side validation: reject invalid model/task combinations
    if request.model_type not in _VALID_MODEL_TASKS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported model type: '{}'. Must be one of: {}".format(
                request.model_type, list(_VALID_MODEL_TASKS.keys())
            ),
        )
    if request.task_type not in _VALID_MODEL_TASKS[request.model_type]:
        raise HTTPException(
            status_code=400,
            detail="Model '{}' does not support task '{}'. Supported tasks: {}".format(
                request.model_type, request.task_type, _VALID_MODEL_TASKS[request.model_type]
            ),
        )

    try:
        # Log the user's choices
        logger.info("Pipeline execution started")
        logger.info("  Model Type: %s", request.model_type)
        logger.info("  Task Type: %s", request.task_type)
        logger.info("  Target Column: %s", request.target_column)
        logger.info("  Dropped Columns: %s", request.dropped_columns)
        logger.info("  Input Data: %s", request.s3_uri)

        # SageMaker requires parameter values to have minimum length 1.
        # Use "none" as a sentinel when the user drops no columns.
        dropped = request.dropped_columns if request.dropped_columns else "none"

        # Start the pipeline execution with user parameters
        response = sagemaker_client.start_pipeline_execution(
            PipelineName=PIPELINE_NAME,
            PipelineParameters=[
                {"Name": "ModelType", "Value": request.model_type},
                {"Name": "TaskType", "Value": request.task_type},
                {"Name": "TargetColumn", "Value": request.target_column},
                {"Name": "DroppedColumns", "Value": dropped},
                {"Name": "InputData", "Value": request.s3_uri},
            ],
            PipelineExecutionDescription="Model: {} | Target: {} | Task: {}".format(
                request.model_type, request.target_column, request.task_type
            ),
        )

        execution_arn = response["PipelineExecutionArn"]
        logger.info("Pipeline execution ARN: %s", execution_arn)

        return TriggerResponse(
            execution_arn=execution_arn,
            status="Executing",
            message="Pipeline triggered successfully with model: {}".format(request.model_type),
        )

    except ClientError as e:
        error_msg = e.response["Error"]["Message"]
        logger.error("Pipeline trigger failed: %s", error_msg)
        raise HTTPException(status_code=500, detail="Pipeline trigger failed: {}".format(error_msg))
    except Exception as e:
        logger.error("Pipeline trigger failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Pipeline trigger failed: {}".format(str(e)))


@app.get("/api/status/{execution_arn:path}", response_model=StatusResponse)
def get_pipeline_status(execution_arn: str):
    """
    Get the status of a pipeline execution.
    Returns the overall status and individual step details.
    """
    try:
        # Describe the pipeline execution
        exec_response = sagemaker_client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )

        status = exec_response.get("PipelineExecutionStatus", "Unknown")
        start_time = str(exec_response.get("CreationTime", ""))
        end_time = str(exec_response.get("LastModifiedTime", ""))

        # Get execution parameters
        parameters = {}
        for param in exec_response.get("PipelineParameters", []):
            parameters[param["Name"]] = param["Value"]

        # Get step details
        steps = []
        try:
            steps_response = sagemaker_client.list_pipeline_execution_steps(
                PipelineExecutionArn=execution_arn,
                SortOrder="Ascending",
            )
            for step in steps_response.get("PipelineExecutionSteps", []):
                step_info = {
                    "name": step.get("StepName", ""),
                    "status": step.get("StepStatus", ""),
                    "start_time": str(step.get("StartTime", "")),
                    "end_time": str(step.get("EndTime", "")),
                    "failure_reason": step.get("FailureReason", ""),
                }
                steps.append(step_info)
        except ClientError:
            # Steps may not be available yet
            pass

        return StatusResponse(
            execution_arn=execution_arn,
            status=status,
            steps=steps,
            start_time=start_time,
            end_time=end_time,
            parameters=parameters,
        )

    except ClientError as e:
        error_msg = e.response["Error"]["Message"]
        raise HTTPException(status_code=404, detail="Execution not found: {}".format(error_msg))


@app.get("/api/artifacts/{execution_arn:path}")
def get_artifacts(execution_arn: str):
    """
    Retrieve artifacts (EDA plots, metrics) for a completed pipeline execution.
    Returns pre-signed S3 URLs for images and parsed JSON for metrics.
    """
    try:
        # First, get the execution details to find the training job
        exec_response = sagemaker_client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )

        if exec_response.get("PipelineExecutionStatus") != "Succeeded":
            return {
                "status": exec_response.get("PipelineExecutionStatus"),
                "message": "Pipeline has not completed successfully yet.",
            }

        # List steps to find the training job
        steps_response = sagemaker_client.list_pipeline_execution_steps(
            PipelineExecutionArn=execution_arn,
        )

        training_job_name = None
        model_package_arn = None

        for step in steps_response.get("PipelineExecutionSteps", []):
            # Find training step
            if step.get("StepName") == "TrainUserSelectedModel":
                metadata = step.get("Metadata", {})
                training_meta = metadata.get("TrainingJob", {})
                training_job_arn = training_meta.get("Arn", "")
                if training_job_arn:
                    training_job_name = training_job_arn.split("/")[-1]

            # Find model registration step.
            # RegisterModel is a StepCollection that produces sub-steps
            # with names like "RegisterTrainedModel-RegisterModel",
            # so we use startswith instead of exact match.
            step_name = step.get("StepName", "")
            if step_name.startswith("RegisterTrainedModel"):
                metadata = step.get("Metadata", {})
                register_meta = metadata.get("RegisterModel", {})
                if register_meta:
                    model_package_arn = register_meta.get("Arn", "")

        if not training_job_name:
            raise HTTPException(
                status_code=404,
                detail="Training job not found in pipeline execution.",
            )

        # Get training job details for output location
        training_response = sagemaker_client.describe_training_job(
            TrainingJobName=training_job_name
        )

        output_path = training_response.get("OutputDataConfig", {}).get("S3OutputPath", "")
        # SageMaker packages output data into output.tar.gz at:
        #   <output_path>/<job_name>/output/output.tar.gz
        output_prefix = "{}/{}/output/".format(
            output_path.replace("s3://{}/".format(S3_BUCKET), ""),
            training_job_name,
        )

        # Extract artifacts from output.tar.gz
        artifacts = _extract_artifacts_from_tar(output_prefix)

        # Add model package ARN if available
        artifacts["model_package_arn"] = model_package_arn or ""

        return artifacts

    except HTTPException:
        raise
    except ClientError as e:
        error_msg = e.response["Error"]["Message"]
        raise HTTPException(status_code=500, detail="Failed to retrieve artifacts: {}".format(error_msg))
    except Exception as e:
        logger.error("Artifact retrieval failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve artifacts: {}".format(str(e)))


@app.get("/api/executions")
def list_executions():
    """List recent pipeline executions for the sidebar history."""
    try:
        response = sagemaker_client.list_pipeline_executions(
            PipelineName=PIPELINE_NAME,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=20,
        )

        executions = []
        for execution in response.get("PipelineExecutionSummaries", []):
            executions.append({
                "execution_arn": execution.get("PipelineExecutionArn", ""),
                "status": execution.get("PipelineExecutionStatus", ""),
                "start_time": str(execution.get("StartTime", "")),
                "description": execution.get("PipelineExecutionDescription", ""),
            })

        return {"executions": executions}

    except ClientError:
        return {"executions": []}


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

def _extract_artifacts_from_tar(output_prefix):
    """
    Download output.tar.gz from S3, extract it in memory, and parse artifacts.
    SageMaker packages all files written to /opt/ml/output/data/ into this tar.
    """
    import tarfile
    import base64

    artifacts = {
        "correlation_heatmap_url": "",
        "missing_value_matrix_url": "",
        "data_summary": {},
        "evaluation_metrics": {},
    }

    tar_key = "{}output.tar.gz".format(output_prefix)
    logger.info("Downloading output.tar.gz from s3://%s/%s", S3_BUCKET, tar_key)

    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=tar_key)
        tar_bytes = response["Body"].read()

        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                logger.info("Found in output.tar.gz: %s", member.name)
                f = tar.extractfile(member)
                if f is None:
                    continue
                content = f.read()

                basename = member.name.split("/")[-1]

                if basename == "evaluation.json":
                    artifacts["evaluation_metrics"] = json.loads(content.decode("utf-8"))
                elif basename == "data_summary.json":
                    artifacts["data_summary"] = json.loads(content.decode("utf-8"))
                elif basename == "correlation_heatmap.png":
                    # Encode image as base64 data URL for the frontend
                    b64 = base64.b64encode(content).decode("utf-8")
                    artifacts["correlation_heatmap_url"] = "data:image/png;base64,{}".format(b64)
                elif basename == "missing_value_matrix.png":
                    b64 = base64.b64encode(content).decode("utf-8")
                    artifacts["missing_value_matrix_url"] = "data:image/png;base64,{}".format(b64)

    except ClientError as e:
        logger.warning("Could not download output.tar.gz: %s", e)
    except Exception as e:
        logger.warning("Error extracting output.tar.gz: %s", e)

    return artifacts
