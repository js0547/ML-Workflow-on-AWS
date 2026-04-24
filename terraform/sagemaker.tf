# ------------------------------------------------------------------
# SageMaker Model Package Group
# Provides model versioning and registry for all trained models.
# Each pipeline execution registers a new model version here.
# ------------------------------------------------------------------

resource "aws_sagemaker_model_package_group" "ml_models" {
  model_package_group_name        = "${var.project_name}-models"
  model_package_group_description = "Model registry for user-selected tabular ML models"

  tags = {
    Project = var.project_name
    Purpose = "Model versioning and registry"
  }
}
