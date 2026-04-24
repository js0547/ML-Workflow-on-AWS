# ------------------------------------------------------------------
# ECS Fargate
# Runs the Streamlit/FastAPI application as a serverless container.
# No server to manage. Pay only for the time the task is running.
# Scale to zero by setting desired_count = 0 when not in use.
# Uses the default VPC with a public IP (no ALB needed for demo).
# ------------------------------------------------------------------

# Use the default VPC -- no need to create networking resources
data "aws_vpc" "default" {
  default = true
}

# Get all default subnets in the region
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Project = var.project_name
  }
}

# Security group for the Fargate task
resource "aws_security_group" "fargate_task" {
  name        = "${var.project_name}-fargate-sg"
  description = "Security group for the ML Platform Fargate task"
  vpc_id      = data.aws_vpc.default.id

  # Streamlit UI
  ingress {
    description = "Streamlit"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # FastAPI backend
  ingress {
    description = "FastAPI"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound traffic (needed for ECR pull, S3, SageMaker API calls)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project = var.project_name
    Name    = "${var.project_name}-fargate-sg"
  }
}

# CloudWatch log group is defined in cloudwatch.tf (aws_cloudwatch_log_group.ecs_app)

# ECS Task Definition
resource "aws_ecs_task_definition" "app" {
  family                   = "${var.project_name}-app"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "ml-platform"
      image     = "${aws_ecr_repository.app.repository_url}:${var.ecr_image_tag}"
      essential = true

      portMappings = [
        {
          containerPort = 8501
          hostPort      = 8501
          protocol      = "tcp"
        },
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "S3_BUCKET"
          value = aws_s3_bucket.ml_data.id
        },
        {
          name  = "SAGEMAKER_ROLE_ARN"
          value = aws_iam_role.sagemaker_execution.arn
        },
        {
          name  = "MODEL_PACKAGE_GROUP"
          value = aws_sagemaker_model_package_group.ml_models.model_package_group_name
        },
        {
          name  = "AWS_DEFAULT_REGION"
          value = var.aws_region
        },
        {
          name  = "PIPELINE_NAME"
          value = "${var.project_name}-pipeline"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_app.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ])

  tags = {
    Project = var.project_name
  }
}

# ECS Service -- runs the task with a public IP
# Set desired_count to 0 when not in use to stop all costs
resource "aws_ecs_service" "app" {
  name            = "${var.project_name}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.fargate_task.id]
    assign_public_ip = true
  }

  # Allow Terraform to proceed even if the image is not yet pushed
  force_new_deployment = true

  tags = {
    Project = var.project_name
  }

  # Ignore desired_count changes so we can scale to zero manually
  lifecycle {
    ignore_changes = [desired_count]
  }
}
