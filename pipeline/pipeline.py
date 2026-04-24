"""
SageMaker Pipeline Definition for the Tabular ML Platform.

Defines a pipeline where ModelType, TargetColumn, DroppedColumns, and TaskType
are ParameterString variables. These parameters are passed from the UI via
FastAPI when the user clicks "Execute Pipeline".

The pipeline uses the SageMaker SKLearn framework container, which supports
installing additional dependencies (xgboost, lightgbm) via requirements.txt
in the source_dir. This avoids the need for a custom training container.

Usage:
    # Upsert the pipeline (run once or on every CI/CD deploy)
    python pipeline.py --upsert

    # Upsert and immediately execute with defaults
    python pipeline.py --upsert --execute
"""

import argparse
import json
import logging
import os
import sys

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import TrainingStep

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# Read from environment variables (set by Terraform/ECS task definition)
# ------------------------------------------------------------------

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-north-1")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")
MODEL_PACKAGE_GROUP = os.environ.get("MODEL_PACKAGE_GROUP", "tabular-ml-platform-models")
PIPELINE_NAME = os.environ.get("PIPELINE_NAME", "tabular-ml-platform-pipeline")

# Training instance configuration
# ml.m5.large is the smallest general-purpose instance suitable for training.
# Instances are ephemeral -- they start, train, and terminate automatically.
TRAINING_INSTANCE_TYPE = "ml.m5.large"
TRAINING_INSTANCE_COUNT = 1

# SKLearn framework version available in SageMaker
SKLEARN_FRAMEWORK_VERSION = "1.2-1"


def get_pipeline_definition(
    role_arn,
    bucket,
    region,
    pipeline_name,
    model_package_group_name,
):
    """
    Build and return the SageMaker Pipeline definition.

    Parameters are defined as ParameterString so they can be overridden
    at execution time from the UI without modifying the pipeline definition.
    """

    # ==============================================================
    # Pipeline Parameters
    # These are the user's choices, passed from the UI at runtime.
    # ==============================================================

    param_model_type = ParameterString(
        name="ModelType",
        default_value="logistic_regression",
    )

    param_task_type = ParameterString(
        name="TaskType",
        default_value="classification",
    )

    param_target_column = ParameterString(
        name="TargetColumn",
        default_value="target",
    )

    param_dropped_columns = ParameterString(
        name="DroppedColumns",
        default_value="",
    )

    param_input_data = ParameterString(
        name="InputData",
        default_value="s3://{}/data/default.csv".format(bucket),
    )

    # ==============================================================
    # Training Step
    # Uses the SageMaker SKLearn framework container.
    # The universal_script.py and requirements.txt are uploaded from
    # the source_dir and installed in the container at runtime.
    # ==============================================================

    # Determine the path to the pipeline source directory
    # This works both when running locally and inside the ECS container
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Retrieve the SKLearn container image URI for the region
    image_uri = sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version=SKLEARN_FRAMEWORK_VERSION,
        py_version="py3",
        instance_type=TRAINING_INSTANCE_TYPE,
    )

    logger.info("Using SKLearn container image: %s", image_uri)

    # Create the estimator
    # Hyperparameters map directly to argparse arguments in universal_script.py
    estimator = Estimator(
        image_uri=image_uri,
        role=role_arn,
        instance_count=TRAINING_INSTANCE_COUNT,
        instance_type=TRAINING_INSTANCE_TYPE,
        entry_point="universal_script.py",
        source_dir=script_dir,
        output_path="s3://{}/pipeline-output".format(bucket),
        base_job_name="{}-training".format(pipeline_name),
        hyperparameters={
            "model_type": param_model_type,
            "task_type": param_task_type,
            "target_column": param_target_column,
            "dropped_columns": param_dropped_columns,
        },
        max_run=3600,
        sagemaker_session=sagemaker.Session(
            boto_session=boto3.Session(region_name=region),
        ),
    )

    # Define the training input from the user-uploaded S3 data
    training_input = TrainingInput(
        s3_data=param_input_data,
        content_type="text/csv",
    )

    # Create the training step
    training_step = TrainingStep(
        name="TrainUserSelectedModel",
        estimator=estimator,
        inputs={"train": training_input},
    )

    # ==============================================================
    # Model Registration Step
    # Registers the trained model in the Model Package Group.
    # Attaches evaluation metrics from the training output.
    # ==============================================================

    # Point to the evaluation metrics JSON produced by universal_script.py
    # The training job saves it to SM_OUTPUT_DATA_DIR which maps to
    # s3://<bucket>/pipeline-output/<job-name>/output/output.tar.gz
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                training_step.properties.OutputDataConfig.S3OutputPath
            ),
            content_type="application/json",
        )
    )

    register_step = RegisterModel(
        name="RegisterTrainedModel",
        estimator=estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.t2.medium"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
        approval_status="PendingManualApproval",
    )

    # ==============================================================
    # Pipeline Definition
    # ==============================================================

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            param_model_type,
            param_task_type,
            param_target_column,
            param_dropped_columns,
            param_input_data,
        ],
        steps=[training_step, register_step],
        sagemaker_session=sagemaker.Session(
            boto_session=boto3.Session(region_name=region),
        ),
    )

    return pipeline


def upsert_pipeline(pipeline):
    """Create or update the pipeline in SageMaker."""
    logger.info("Upserting pipeline: %s", pipeline.name)
    response = pipeline.upsert(role_arn=SAGEMAKER_ROLE_ARN)
    logger.info("Pipeline ARN: %s", response["PipelineArn"])
    return response


def execute_pipeline(pipeline, parameters=None):
    """Start a pipeline execution with optional parameter overrides."""
    logger.info("Starting pipeline execution...")

    execution_params = {}
    if parameters:
        execution_params["PipelineParameters"] = parameters

    execution = pipeline.start(**execution_params)
    logger.info("Execution ARN: %s", execution.arn)
    return execution


def describe_pipeline(pipeline_name, region=AWS_REGION):
    """Describe an existing pipeline to verify it exists."""
    sm_client = boto3.client("sagemaker", region_name=region)
    try:
        response = sm_client.describe_pipeline(PipelineName=pipeline_name)
        logger.info("Pipeline: %s", response["PipelineName"])
        logger.info("Status: %s", response["PipelineStatus"])
        logger.info("Created: %s", response["CreationTime"])
        logger.info("Modified: %s", response["LastModifiedTime"])
        return response
    except sm_client.exceptions.ResourceNotFound:
        logger.warning("Pipeline '%s' does not exist.", pipeline_name)
        return None


# ------------------------------------------------------------------
# CLI Entry Point
# Used for manual pipeline management and CI/CD upsert.
# ------------------------------------------------------------------

def main():
    """CLI entry point for pipeline management."""
    parser = argparse.ArgumentParser(
        description="Manage the SageMaker Pipeline for the Tabular ML Platform."
    )
    parser.add_argument(
        "--upsert",
        action="store_true",
        help="Create or update the pipeline definition in SageMaker.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the pipeline after upserting (uses default parameters).",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Describe the current pipeline status.",
    )
    parser.add_argument(
        "--role_arn",
        type=str,
        default=SAGEMAKER_ROLE_ARN,
        help="SageMaker execution role ARN (overrides env var).",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=S3_BUCKET,
        help="S3 bucket name (overrides env var).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=AWS_REGION,
        help="AWS region (overrides env var).",
    )

    args = parser.parse_args()

    # Validate required configuration
    role_arn = args.role_arn
    bucket = args.bucket
    region = args.region

    if not role_arn:
        logger.error("SageMaker role ARN is required. Set SAGEMAKER_ROLE_ARN or use --role_arn.")
        sys.exit(1)
    if not bucket:
        logger.error("S3 bucket is required. Set S3_BUCKET or use --bucket.")
        sys.exit(1)

    if args.describe:
        describe_pipeline(PIPELINE_NAME, region)
        return

    if args.upsert:
        pipeline = get_pipeline_definition(
            role_arn=role_arn,
            bucket=bucket,
            region=region,
            pipeline_name=PIPELINE_NAME,
            model_package_group_name=MODEL_PACKAGE_GROUP,
        )
        upsert_pipeline(pipeline)

        if args.execute:
            execute_pipeline(pipeline)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
