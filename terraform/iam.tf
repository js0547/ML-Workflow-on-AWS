# ------------------------------------------------------------------
# IAM Roles and Policies
# Least-privilege access for SageMaker and ECS Fargate components.
# ------------------------------------------------------------------

# ==================================================================
# SageMaker Execution Role
# Used by SageMaker training jobs, processing jobs, and pipelines.
# ==================================================================

resource "aws_iam_role" "sagemaker_execution" {
  name = "${var.project_name}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project = var.project_name
    Purpose = "SageMaker execution role"
  }
}

# S3 access scoped to the project bucket only
resource "aws_iam_role_policy" "sagemaker_s3" {
  name = "${var.project_name}-sagemaker-s3"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.ml_data.arn,
          "${aws_s3_bucket.ml_data.arn}/*"
        ]
      }
    ]
  })
}

# SageMaker permissions for training, processing, and model registry
resource "aws_iam_role_policy" "sagemaker_core" {
  name = "${var.project_name}-sagemaker-core"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:CreateProcessingJob",
          "sagemaker:DescribeProcessingJob",
          "sagemaker:CreateModel",
          "sagemaker:DescribeModel",
          "sagemaker:CreateModelPackage",
          "sagemaker:CreateModelPackageGroup",
          "sagemaker:DescribeModelPackage",
          "sagemaker:DescribeModelPackageGroup",
          "sagemaker:UpdateModelPackage",
          "sagemaker:ListModelPackages",
          "sagemaker:AddTags"
        ]
        Resource = "*"
      }
    ]
  })
}

# CloudWatch Logs for training job output
resource "aws_iam_role_policy" "sagemaker_logs" {
  name = "${var.project_name}-sagemaker-logs"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*"
      }
    ]
  })
}

# ECR pull access for SageMaker to use framework containers
resource "aws_iam_role_policy" "sagemaker_ecr" {
  name = "${var.project_name}-sagemaker-ecr"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetAuthorizationToken"
        ]
        Resource = "*"
      }
    ]
  })
}

# PassRole so SageMaker can pass its own role to training jobs
resource "aws_iam_role_policy" "sagemaker_passrole" {
  name = "${var.project_name}-sagemaker-passrole"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = aws_iam_role.sagemaker_execution.arn
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "sagemaker.amazonaws.com"
          }
        }
      }
    ]
  })
}

# ==================================================================
# ECS Task Execution Role
# Used by ECS/Fargate to pull images from ECR and write logs.
# This is the "infrastructure" role, not the application role.
# ==================================================================

resource "aws_iam_role" "ecs_task_execution" {
  name = "${var.project_name}-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project = var.project_name
    Purpose = "ECS task execution role - pull images and write logs"
  }
}

# Attach the AWS managed policy for ECS task execution
resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ==================================================================
# ECS Task Role
# Used by the application code running inside the container.
# Grants access to S3, SageMaker pipeline trigger, etc.
# ==================================================================

resource "aws_iam_role" "ecs_task" {
  name = "${var.project_name}-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project = var.project_name
    Purpose = "ECS task role - application permissions"
  }
}

# S3 access for uploading data and reading artifacts
resource "aws_iam_role_policy" "ecs_task_s3" {
  name = "${var.project_name}-ecs-task-s3"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.ml_data.arn,
          "${aws_s3_bucket.ml_data.arn}/*"
        ]
      }
    ]
  })
}

# SageMaker pipeline trigger and monitoring permissions
resource "aws_iam_role_policy" "ecs_task_sagemaker" {
  name = "${var.project_name}-ecs-task-sagemaker"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:StartPipelineExecution",
          "sagemaker:DescribePipelineExecution",
          "sagemaker:ListPipelineExecutionSteps",
          "sagemaker:DescribePipeline",
          "sagemaker:ListPipelineExecutions",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:DescribeModelPackage",
          "sagemaker:ListModelPackages"
        ]
        Resource = "*"
      }
    ]
  })
}

# CloudWatch permissions for application logging and custom metrics
resource "aws_iam_role_policy" "ecs_task_cloudwatch" {
  name = "${var.project_name}-ecs-task-cloudwatch"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/ecs/${var.project_name}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = "${var.project_name}/Application"
          }
        }
      }
    ]
  })
}

