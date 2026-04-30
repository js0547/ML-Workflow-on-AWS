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
import tempfile
import time
from datetime import datetime

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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


# ------------------------------------------------------------------
# Custom Script Endpoints
# ------------------------------------------------------------------

SKLEARN_FRAMEWORK_VERSION = "1.2-1"
CUSTOM_TRAINING_INSTANCE = "ml.m5.large"


@app.post("/api/custom/upload")
async def upload_custom_files(
    script: UploadFile = File(...),
    dataset: UploadFile = File(...),
    requirements: UploadFile = File(None),
):
    """
    Upload a custom Python training script, dataset CSV, and optional
    requirements.txt to S3. Returns S3 keys for use in /api/custom/trigger.
    """
    if not script.filename.endswith(".py"):
        raise HTTPException(status_code=400, detail="Script must be a .py file.")
    if not dataset.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Dataset must be a .csv file.")

    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Upload script
        script_content = await script.read()
        script_key = "custom/scripts/{}_{}".format(timestamp, script.filename)
        s3_client.put_object(Bucket=S3_BUCKET, Key=script_key, Body=script_content)

        # Upload dataset
        dataset_content = await dataset.read()
        dataset_key = "custom/data/{}_{}".format(timestamp, dataset.filename)
        s3_client.put_object(Bucket=S3_BUCKET, Key=dataset_key, Body=dataset_content, ContentType="text/csv")

        # Upload optional requirements
        req_key = ""
        if requirements and requirements.filename:
            req_content = await requirements.read()
            req_key = "custom/scripts/{}_requirements.txt".format(timestamp)
            s3_client.put_object(Bucket=S3_BUCKET, Key=req_key, Body=req_content)

        row_count = 0
        col_count = 0
        try:
            df = pd.read_csv(io.BytesIO(dataset_content))
            row_count = len(df)
            col_count = len(df.columns)
        except Exception:
            pass

        logger.info("Custom upload: script=%s, dataset=%s (%d rows)",
                     script.filename, dataset.filename, row_count)

        return {
            "script_s3_key": script_key,
            "script_filename": script.filename,
            "requirements_s3_key": req_key,
            "dataset_s3_uri": "s3://{}/{}".format(S3_BUCKET, dataset_key),
            "dataset_filename": dataset.filename,
            "row_count": row_count,
            "column_count": col_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Custom upload failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Upload failed: {}".format(str(e)))


@app.post("/api/custom/trigger")
def trigger_custom_job(
    script_s3_key: str = Form(...),
    script_filename: str = Form(...),
    dataset_s3_uri: str = Form(...),
    requirements_s3_key: str = Form(""),
):
    """
    Start a SageMaker training job using the user's custom script.
    Downloads the script to a temp directory and uses the SageMaker SDK
    to submit a training job on an ephemeral ml.m5.large instance.
    """
    tmpdir = None
    try:
        # Stage script files locally so the SageMaker SDK can package them
        tmpdir = tempfile.mkdtemp(prefix="custom_script_")

        script_path = os.path.join(tmpdir, script_filename)
        s3_client.download_file(S3_BUCKET, script_s3_key, script_path)

        if requirements_s3_key:
            req_path = os.path.join(tmpdir, "requirements.txt")
            s3_client.download_file(S3_BUCKET, requirements_s3_key, req_path)

        logger.info("Custom job: script=%s, data=%s", script_filename, dataset_s3_uri)

        # Lazy import sagemaker SDK (handles both v2 and v3 import paths)
        import sagemaker
        try:
            from sagemaker.estimator import Estimator as _Estimator
        except ImportError:
            from sagemaker import Estimator as _Estimator
        try:
            from sagemaker.inputs import TrainingInput as _TrainingInput
        except ImportError:
            from sagemaker import TrainingInput as _TrainingInput

        image_uri = sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=AWS_REGION,
            version=SKLEARN_FRAMEWORK_VERSION,
            py_version="py3",
            instance_type=CUSTOM_TRAINING_INSTANCE,
        )

        sm_session = sagemaker.Session(
            boto_session=boto3.Session(region_name=AWS_REGION),
        )

        estimator = _Estimator(
            image_uri=image_uri,
            role=SAGEMAKER_ROLE_ARN,
            instance_count=1,
            instance_type=CUSTOM_TRAINING_INSTANCE,
            entry_point=script_filename,
            source_dir=tmpdir,
            output_path="s3://{}/custom-output".format(S3_BUCKET),
            base_job_name="custom-script",
            max_run=7200,
            sagemaker_session=sm_session,
        )

        training_input = _TrainingInput(
            s3_data=dataset_s3_uri,
            content_type="text/csv",
        )

        estimator.fit({"train": training_input}, wait=False)
        job_name = estimator.latest_training_job.name

        logger.info("Custom training job started: %s", job_name)
        return {
            "job_name": job_name,
            "status": "InProgress",
            "message": "Custom training job started: {}".format(job_name),
        }

    except ClientError as e:
        error_msg = e.response["Error"]["Message"]
        logger.error("Custom job trigger failed: %s", error_msg)
        raise HTTPException(status_code=500, detail="Trigger failed: {}".format(error_msg))
    except Exception as e:
        logger.error("Custom job trigger failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Trigger failed: {}".format(str(e)))
    finally:
        # Clean up temp directory (SDK has already uploaded source to S3)
        if tmpdir and os.path.exists(tmpdir):
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/api/custom/status/{job_name}")
def get_custom_job_status(job_name: str):
    """Get the status of a custom training job."""
    try:
        resp = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        status = resp.get("TrainingJobStatus", "Unknown")
        secondary = resp.get("SecondaryStatus", "")
        failure = resp.get("FailureReason", "")
        start_time = str(resp.get("TrainingStartTime", ""))
        end_time = str(resp.get("TrainingEndTime", ""))
        duration = ""
        if resp.get("TrainingStartTime") and resp.get("TrainingEndTime"):
            delta = resp["TrainingEndTime"] - resp["TrainingStartTime"]
            duration = str(int(delta.total_seconds())) + "s"

        return {
            "job_name": job_name,
            "status": status,
            "secondary_status": secondary,
            "failure_reason": failure,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
        }
    except ClientError as e:
        raise HTTPException(status_code=404, detail="Job not found: {}".format(
            e.response["Error"]["Message"]))


@app.get("/api/custom/artifacts/{job_name}")
def get_custom_job_artifacts(job_name: str):
    """Retrieve output artifacts from a completed custom training job."""
    try:
        resp = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        if resp.get("TrainingJobStatus") != "Completed":
            return {
                "status": resp.get("TrainingJobStatus"),
                "message": "Job has not completed yet.",
            }

        output_path = resp.get("OutputDataConfig", {}).get("S3OutputPath", "")
        output_prefix = "{}/{}/output/".format(
            output_path.replace("s3://{}/".format(S3_BUCKET), ""),
            job_name,
        )
        return _extract_artifacts_from_tar(output_prefix)

    except ClientError as e:
        raise HTTPException(status_code=500, detail="Artifact retrieval failed: {}".format(
            e.response["Error"]["Message"]))


@app.get("/api/custom/jobs")
def list_custom_jobs():
    """List recent custom training jobs."""
    try:
        resp = sagemaker_client.list_training_jobs(
            NameContains="custom-script",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=20,
        )
        jobs = []
        for job in resp.get("TrainingJobSummaries", []):
            jobs.append({
                "job_name": job.get("TrainingJobName", ""),
                "status": job.get("TrainingJobStatus", ""),
                "start_time": str(job.get("CreationTime", "")),
            })
        return {"jobs": jobs}
    except ClientError:
        return {"jobs": []}
