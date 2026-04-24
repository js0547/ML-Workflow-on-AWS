# ------------------------------------------------------------------
# Terraform Configuration
# Provider setup and backend configuration for the ML Platform
# ------------------------------------------------------------------

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ------------------------------------------------------------------
# Data Sources
# Used to dynamically reference the current AWS account and region
# ------------------------------------------------------------------

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
