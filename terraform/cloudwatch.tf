# ------------------------------------------------------------------
# CloudWatch Monitoring and Logging
# Centralized logging for ECS and SageMaker.
# Alarms for cost protection and failure detection.
# Dashboard for a single-pane-of-glass view.
# ------------------------------------------------------------------

# ==================================================================
# Log Groups
# ==================================================================

# ECS application logs (Streamlit + FastAPI)
# Already referenced in ecs.tf container definition
resource "aws_cloudwatch_log_group" "ecs_app" {
  name              = "/ecs/${var.project_name}/app"
  retention_in_days = 14

  tags = {
    Project   = var.project_name
    Component = "ecs-app"
  }
}

# SageMaker training job logs
resource "aws_cloudwatch_log_group" "sagemaker_training" {
  name              = "/aws/sagemaker/TrainingJobs/${var.project_name}"
  retention_in_days = 14

  tags = {
    Project   = var.project_name
    Component = "sagemaker-training"
  }
}

# SageMaker pipeline execution logs
resource "aws_cloudwatch_log_group" "sagemaker_pipeline" {
  name              = "/aws/sagemaker/Pipelines/${var.project_name}"
  retention_in_days = 14

  tags = {
    Project   = var.project_name
    Component = "sagemaker-pipeline"
  }
}

# ==================================================================
# SNS Topic for Alarm Notifications
# ==================================================================

resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"

  tags = {
    Project = var.project_name
  }
}

# Subscribe an email address for alarm notifications
resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ==================================================================
# Alarms
# ==================================================================

# Alert if ECS service CPU exceeds 80 percent
resource "aws_cloudwatch_metric_alarm" "ecs_cpu_high" {
  alarm_name          = "${var.project_name}-ecs-cpu-high"
  alarm_description   = "ECS Fargate task CPU utilization exceeded 80 percent"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  treat_missing_data  = "notBreaching"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.app.name
  }

  alarm_actions = [aws_sns_topic.alerts.arn]

  tags = {
    Project = var.project_name
  }
}

# Alert if ECS service memory exceeds 80 percent
resource "aws_cloudwatch_metric_alarm" "ecs_memory_high" {
  alarm_name          = "${var.project_name}-ecs-memory-high"
  alarm_description   = "ECS Fargate task memory utilization exceeded 80 percent"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "MemoryUtilization"
  namespace           = "AWS/ECS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  treat_missing_data  = "notBreaching"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.app.name
  }

  alarm_actions = [aws_sns_topic.alerts.arn]

  tags = {
    Project = var.project_name
  }
}

# Alert on SageMaker training job failures via error log activity
resource "aws_cloudwatch_metric_alarm" "sagemaker_errors" {
  alarm_name          = "${var.project_name}-sagemaker-errors"
  alarm_description   = "SageMaker training job errors detected in logs"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "TrainingJobErrors"
  namespace           = "${var.project_name}/SageMaker"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  treat_missing_data  = "notBreaching"

  alarm_actions = [aws_sns_topic.alerts.arn]

  tags = {
    Project = var.project_name
  }
}

# ==================================================================
# Metric Filters
# Extract custom metrics from log streams for alarming
# ==================================================================

# Count ERROR level log entries from the application
resource "aws_cloudwatch_log_metric_filter" "app_errors" {
  name           = "${var.project_name}-app-errors"
  log_group_name = aws_cloudwatch_log_group.ecs_app.name
  pattern        = "ERROR"

  metric_transformation {
    name          = "ApplicationErrors"
    namespace     = "${var.project_name}/Application"
    value         = "1"
    default_value = "0"
  }
}

# Count pipeline trigger events from FastAPI logs
resource "aws_cloudwatch_log_metric_filter" "pipeline_triggers" {
  name           = "${var.project_name}-pipeline-triggers"
  log_group_name = aws_cloudwatch_log_group.ecs_app.name
  pattern        = "\"Pipeline execution started\""

  metric_transformation {
    name          = "PipelineTriggers"
    namespace     = "${var.project_name}/Application"
    value         = "1"
    default_value = "0"
  }
}

# ==================================================================
# Dashboard
# Single view of the entire platform health
# ==================================================================

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "text"
        x      = 0
        y      = 0
        width  = 24
        height = 1
        properties = {
          markdown = "# Tabular ML Platform -- Monitoring Dashboard"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 1
        width  = 12
        height = 6
        properties = {
          title   = "ECS CPU Utilization"
          metrics = [
            ["AWS/ECS", "CPUUtilization", "ClusterName", aws_ecs_cluster.main.name, "ServiceName", aws_ecs_service.app.name]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          view   = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 1
        width  = 12
        height = 6
        properties = {
          title   = "ECS Memory Utilization"
          metrics = [
            ["AWS/ECS", "MemoryUtilization", "ClusterName", aws_ecs_cluster.main.name, "ServiceName", aws_ecs_service.app.name]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          view   = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 7
        width  = 12
        height = 6
        properties = {
          title   = "Application Errors"
          metrics = [
            ["${var.project_name}/Application", "ApplicationErrors"]
          ]
          period = 300
          stat   = "Sum"
          region = var.aws_region
          view   = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 7
        width  = 12
        height = 6
        properties = {
          title   = "Pipeline Executions Triggered"
          metrics = [
            ["${var.project_name}/Application", "PipelineTriggers"]
          ]
          period = 300
          stat   = "Sum"
          region = var.aws_region
          view   = "timeSeries"
        }
      },
      {
        type   = "log"
        x      = 0
        y      = 13
        width  = 24
        height = 6
        properties = {
          title   = "Recent Application Logs"
          query   = "SOURCE '${aws_cloudwatch_log_group.ecs_app.name}' | fields @timestamp, @message | sort @timestamp desc | limit 50"
          region  = var.aws_region
          view    = "table"
        }
      },
      {
        type   = "log"
        x      = 0
        y      = 19
        width  = 24
        height = 6
        properties = {
          title   = "Recent SageMaker Training Logs"
          query   = "SOURCE '${aws_cloudwatch_log_group.sagemaker_training.name}' | fields @timestamp, @message | sort @timestamp desc | limit 50"
          region  = var.aws_region
          view    = "table"
        }
      }
    ]
  })
}
