"""
Streamlit UI for the Tabular ML Platform.

The user makes all key modeling decisions through this interface:
1. Upload a CSV dataset
2. Select the target column
3. Choose columns to drop
4. Select the ML model
5. Select the task type (classification/regression)
6. Execute the pipeline and view results

No Auto-ML -- every choice is made by the user.
"""

import json
import time

import requests
import streamlit as st

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# FastAPI backend URL (running in the same container)
API_BASE_URL = "http://localhost:8000"

# Available models and their supported task types
MODEL_OPTIONS = {
    "Logistic Regression": {
        "value": "logistic_regression",
        "tasks": ["classification"],
        "library": "scikit-learn",
    },
    "Linear Regression": {
        "value": "linear_regression",
        "tasks": ["regression"],
        "library": "scikit-learn",
    },
    "XGBoost": {
        "value": "xgboost",
        "tasks": ["classification", "regression"],
        "library": "xgboost",
    },
    "LightGBM": {
        "value": "lightgbm",
        "tasks": ["classification", "regression"],
        "library": "lightgbm",
    },
}

# ------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Tabular ML Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS for clean, professional styling
# ------------------------------------------------------------------

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Header styling */
    .platform-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .platform-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: white;
    }
    .platform-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.85;
        font-size: 1.05rem;
        color: #e0e0e0;
    }

    /* Card sections */
    .section-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .section-card h3 {
        margin-top: 0;
        color: #1a1a2e;
        font-size: 1.15rem;
        border-bottom: 2px solid #0f3460;
        padding-bottom: 0.5rem;
    }

    /* Status badges */
    .status-executing {
        background: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-succeeded {
        background: #d4edda;
        color: #155724;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-failed {
        background: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f3460;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Step indicator */
    .step-indicator {
        display: inline-block;
        background: #0f3460;
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        text-align: center;
        line-height: 28px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------

if "upload_result" not in st.session_state:
    st.session_state.upload_result = None
if "execution_arn" not in st.session_state:
    st.session_state.execution_arn = None
if "pipeline_status" not in st.session_state:
    st.session_state.pipeline_status = None
if "artifacts" not in st.session_state:
    st.session_state.artifacts = None

# Custom script session state
if "custom_upload" not in st.session_state:
    st.session_state.custom_upload = None
if "custom_job_name" not in st.session_state:
    st.session_state.custom_job_name = None
if "custom_artifacts" not in st.session_state:
    st.session_state.custom_artifacts = None


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

def api_request(method, endpoint, **kwargs):
    """Make a request to the FastAPI backend."""
    url = "{}{}".format(API_BASE_URL, endpoint)
    try:
        response = requests.request(method, url, timeout=60, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API backend. Make sure FastAPI is running on port 8000.")
        return None
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error("API Error: {}".format(detail))
        return None
    except Exception as e:
        st.error("Request failed: {}".format(str(e)))
        return None


def get_status_html(status):
    """Return an HTML badge for the pipeline status."""
    status_lower = status.lower()
    if status_lower in ("executing", "starting"):
        return '<span class="status-executing">{}</span>'.format(status)
    elif status_lower == "succeeded":
        return '<span class="status-succeeded">{}</span>'.format(status)
    elif status_lower in ("failed", "stopped"):
        return '<span class="status-failed">{}</span>'.format(status)
    return '<span>{}</span>'.format(status)


# ------------------------------------------------------------------
# Sidebar: Pipeline Execution History
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Pipeline History")
    st.markdown("---")

    history = api_request("GET", "/api/executions")
    if history and history.get("executions"):
        for exec_info in history["executions"]:
            status = exec_info.get("status", "Unknown")
            desc = exec_info.get("description", "No description")
            start = exec_info.get("start_time", "")

            # Truncate the ARN for display
            arn = exec_info.get("execution_arn", "")
            short_id = arn.split("/")[-1][:12] if "/" in arn else arn[:12]

            with st.expander("{} - {}".format(short_id, status)):
                st.text("Status: {}".format(status))
                st.text("Started: {}".format(start[:19] if start else "N/A"))
                st.text("Config: {}".format(desc))
                if st.button("View Results", key="view_{}".format(short_id)):
                    st.session_state.execution_arn = arn
                    st.rerun()
    else:
        st.info("No pipeline executions yet. Upload a dataset and run your first pipeline.")

    st.markdown("---")
    st.markdown("### Platform Info")
    health = api_request("GET", "/api/health")
    if health:
        st.text("Region: {}".format(health.get("region", "N/A")))
        st.text("Bucket: {}".format(health.get("bucket", "N/A")))
        st.text("Pipeline: {}".format(health.get("pipeline", "N/A")))


# ------------------------------------------------------------------
# Main Content
# ------------------------------------------------------------------

# Header
st.markdown("""
<div class="platform-header">
    <h1>Tabular ML Platform</h1>
    <p>Upload your data, choose your model, and execute -- every decision is yours.</p>
</div>
""", unsafe_allow_html=True)

# Mode selector
workflow_mode = st.radio(
    "Workflow Mode",
    ["Built-in Models", "Custom Script"],
    horizontal=True,
    help="Built-in Models: select from supported models. Custom Script: upload your own training script.",
)

# ==================================================================
# CUSTOM SCRIPT MODE
# ==================================================================

if workflow_mode == "Custom Script":

    # -- Info box with SageMaker conventions --
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h3><span class="step-indicator">i</span> Script Requirements</h3>', unsafe_allow_html=True)
    st.markdown(
        "Your script will run inside a **SageMaker SKLearn 1.2 container** "
        "(Python 3.9). Follow these conventions:\n"
        "- Training data directory: use `argparse` arg `--train` or env var `SM_CHANNEL_TRAIN`\n"
        "- Save your model to: `--model_dir` or env var `SM_MODEL_DIR` (`/opt/ml/model`)\n"
        "- Save other outputs (metrics, plots) to: `--output_dir` or env var `SM_OUTPUT_DATA_DIR` (`/opt/ml/output/data`)\n"
        "- Pre-installed packages: scikit-learn, pandas, numpy, scipy, joblib\n"
        "- Add extra dependencies via an optional `requirements.txt` upload"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # -- Step 1: Upload files --
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h3><span class="step-indicator">1</span> Upload Script & Data</h3>', unsafe_allow_html=True)

    col_s, col_d = st.columns(2)
    with col_s:
        custom_script = st.file_uploader(
            "Training Script (.py)",
            type=["py"],
            help="Your Python training script.",
        )
    with col_d:
        custom_dataset = st.file_uploader(
            "Dataset (.csv)",
            type=["csv"],
            key="custom_csv",
            help="CSV dataset to train on.",
        )

    custom_requirements = st.file_uploader(
        "requirements.txt (optional)",
        type=["txt"],
        help="Extra pip dependencies to install before training.",
    )

    if custom_script and custom_dataset:
        upload_needed = (
            st.session_state.custom_upload is None
            or st.session_state.custom_upload.get("script_filename") != custom_script.name
            or st.session_state.custom_upload.get("dataset_filename") != custom_dataset.name
        )
        if upload_needed:
            with st.spinner("Uploading files to S3..."):
                files = {
                    "script": (custom_script.name, custom_script.getvalue(), "text/x-python"),
                    "dataset": (custom_dataset.name, custom_dataset.getvalue(), "text/csv"),
                }
                if custom_requirements:
                    files["requirements"] = (
                        custom_requirements.name,
                        custom_requirements.getvalue(),
                        "text/plain",
                    )
                result = api_request("POST", "/api/custom/upload", files=files)
                if result:
                    st.session_state.custom_upload = result
                    st.success(
                        "Uploaded: {} + {} ({} rows, {} cols)".format(
                            result["script_filename"],
                            result["dataset_filename"],
                            result["row_count"],
                            result["column_count"],
                        )
                    )

    st.markdown('</div>', unsafe_allow_html=True)

    # -- Step 2: Trigger --
    if st.session_state.custom_upload:
        upload = st.session_state.custom_upload

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3><span class="step-indicator">2</span> Run Training Job</h3>', unsafe_allow_html=True)

        st.info(
            "Script **{}** will run on an ephemeral **ml.m5.large** instance. "
            "The instance terminates automatically after training.".format(upload["script_filename"])
        )

        if st.button("Start Training Job", type="primary", use_container_width=True, key="custom_trigger"):
            with st.spinner("Starting SageMaker training job..."):
                trigger_data = {
                    "script_s3_key": upload["script_s3_key"],
                    "script_filename": upload["script_filename"],
                    "dataset_s3_uri": upload["dataset_s3_uri"],
                    "requirements_s3_key": upload.get("requirements_s3_key", ""),
                }
                trigger_result = api_request("POST", "/api/custom/trigger", data=trigger_data)
                if trigger_result:
                    st.session_state.custom_job_name = trigger_result["job_name"]
                    st.session_state.custom_artifacts = None
                    st.success(trigger_result["message"])

        st.markdown('</div>', unsafe_allow_html=True)

    # -- Step 3: Status & Artifacts --
    if st.session_state.custom_job_name:
        job_name = st.session_state.custom_job_name

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3><span class="step-indicator">3</span> Job Status & Results</h3>', unsafe_allow_html=True)

        status_data = api_request("GET", "/api/custom/status/{}".format(job_name))
        if status_data:
            current_status = status_data.get("status", "Unknown")
            secondary = status_data.get("secondary_status", "")

            st.markdown("**Job:** `{}`".format(job_name))
            st.markdown("**Status:** {} {}".format(current_status, "-- {}".format(secondary) if secondary else ""))

            if status_data.get("duration"):
                st.markdown("**Duration:** {}".format(status_data["duration"]))

            if status_data.get("failure_reason"):
                st.error("Failure: {}".format(status_data["failure_reason"]))

            # Auto-refresh while in progress
            if current_status == "InProgress":
                st.info("Training job is running. Page will refresh automatically.")
                time.sleep(15)
                st.rerun()

            # Show artifacts when completed
            if current_status == "Completed":
                st.success("Training job completed successfully.")

                if st.session_state.custom_artifacts is None:
                    with st.spinner("Loading output artifacts..."):
                        artifacts = api_request("GET", "/api/custom/artifacts/{}".format(job_name))
                        if artifacts and "message" not in artifacts:
                            st.session_state.custom_artifacts = artifacts

                if st.session_state.custom_artifacts:
                    arts = st.session_state.custom_artifacts

                    # Show any images found
                    img_col1, img_col2 = st.columns(2)
                    with img_col1:
                        if arts.get("correlation_heatmap_url"):
                            st.markdown("**Correlation Heatmap**")
                            st.image(arts["correlation_heatmap_url"], use_container_width=True)
                    with img_col2:
                        if arts.get("missing_value_matrix_url"):
                            st.markdown("**Missing Value Matrix**")
                            st.image(arts["missing_value_matrix_url"], use_container_width=True)

                    # Show metrics if available
                    metrics = arts.get("evaluation_metrics", {})
                    if metrics:
                        st.markdown("---")
                        st.markdown("#### Output Metrics")
                        st.json(metrics)

                    # Show data summary if available
                    summary = arts.get("data_summary", {})
                    if summary:
                        with st.expander("Data Summary"):
                            st.json(summary)

                    if not any([arts.get("correlation_heatmap_url"),
                                arts.get("missing_value_matrix_url"),
                                arts.get("evaluation_metrics"),
                                arts.get("data_summary")]):
                        st.info(
                            "No standard artifacts found in output.tar.gz. "
                            "Your model was saved to S3 successfully."
                        )

        st.markdown('</div>', unsafe_allow_html=True)

    # -- Custom job history in sidebar is handled by the existing sidebar code --

    st.markdown("---")
    st.caption(
        "Custom Script Mode -- Your script runs as-is on SageMaker. "
        "The platform provides infrastructure; you provide the logic."
    )
    st.stop()  # Prevent built-in model UI from rendering below


# ==================================================================
# STEP 1: Data Upload
# ==================================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<h3><span class="step-indicator">1</span> Upload Dataset</h3>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload the tabular dataset you want to train a model on.",
)

if uploaded_file is not None:
    # Upload to S3 via FastAPI
    if st.session_state.upload_result is None or st.session_state.upload_result.get("filename") != uploaded_file.name:
        with st.spinner("Uploading to S3..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            result = api_request("POST", "/api/upload", files=files)
            if result:
                st.session_state.upload_result = result
                st.success("Uploaded: {} ({} rows, {} columns)".format(
                    result["filename"], result["row_count"], result["column_count"]
                ))

    # Show dataset preview
    if st.session_state.upload_result:
        result = st.session_state.upload_result
        with st.expander("Dataset Preview", expanded=False):
            import pandas as pd
            preview_df = pd.read_csv(uploaded_file)
            st.dataframe(preview_df.head(10), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# ==================================================================
# STEP 2: The Selector -- User Configuration
# ==================================================================

if st.session_state.upload_result:
    result = st.session_state.upload_result
    columns = result["columns"]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h3><span class="step-indicator">2</span> Configure Model</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Target Column Selector
        target_column = st.selectbox(
            "Target Column",
            options=columns,
            help="Select the column you want to predict.",
            index=len(columns) - 1,
        )

        # Task Type Selector
        task_type = st.radio(
            "Task Type",
            options=["Classification", "Regression"],
            help="Classification for categorical targets, Regression for numeric targets.",
            horizontal=True,
        )

    with col2:
        # Columns to Drop
        available_for_drop = [c for c in columns if c != target_column]
        dropped_columns = st.multiselect(
            "Columns to Drop",
            options=available_for_drop,
            help="Select columns to exclude from training (e.g., IDs, timestamps).",
        )

        # Model Selector -- filtered by task type
        task_type_lower = task_type.lower()
        available_models = {
            name: info for name, info in MODEL_OPTIONS.items()
            if task_type_lower in info["tasks"]
        }

        model_name = st.selectbox(
            "ML Model",
            options=list(available_models.keys()),
            help="Select the model to train. Your choice is final -- no auto-selection.",
        )

    # Summary Card
    st.markdown("---")
    st.markdown("**Configuration Summary:**")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    with summary_col1:
        st.metric("Target", target_column)
    with summary_col2:
        st.metric("Task Type", task_type)
    with summary_col3:
        st.metric("Model", model_name)
    with summary_col4:
        st.metric("Dropped Cols", str(len(dropped_columns)))

    model_info = available_models[model_name]
    st.caption("Library: {} | Model code: {}".format(
        model_info["library"], model_info["value"]
    ))

    st.markdown('</div>', unsafe_allow_html=True)


    # ==============================================================
    # STEP 3: The Trigger -- Execute Pipeline
    # ==============================================================

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h3><span class="step-indicator">3</span> Execute Pipeline</h3>', unsafe_allow_html=True)

    st.info(
        "This will start a SageMaker Pipeline with your exact configuration. "
        "The training job runs on an ephemeral ml.m5.large instance that "
        "terminates automatically after training."
    )

    execute_button = st.button(
        "Execute Pipeline",
        type="primary",
        use_container_width=True,
    )

    if execute_button:
        with st.spinner("Triggering SageMaker Pipeline..."):
            trigger_data = {
                "s3_uri": result["s3_uri"],
                "target_column": target_column,
                "dropped_columns": ",".join(dropped_columns),
                "model_type": model_info["value"],
                "task_type": task_type_lower,
            }
            trigger_result = api_request("POST", "/api/trigger", json=trigger_data)
            if trigger_result:
                st.session_state.execution_arn = trigger_result["execution_arn"]
                st.session_state.pipeline_status = "Executing"
                st.session_state.artifacts = None
                st.success("Pipeline triggered: {}".format(trigger_result["message"]))

    st.markdown('</div>', unsafe_allow_html=True)


# ==================================================================
# STEP 4: Pipeline Status and Results
# ==================================================================

if st.session_state.execution_arn:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h3><span class="step-indicator">4</span> Pipeline Status and Results</h3>', unsafe_allow_html=True)

    # Status display
    execution_arn = st.session_state.execution_arn

    # Poll for status
    status_data = api_request("GET", "/api/status/{}".format(execution_arn))

    if status_data:
        current_status = status_data.get("status", "Unknown")
        st.session_state.pipeline_status = current_status

        # Status header
        st.markdown(
            "**Execution Status:** {}".format(current_status),
            unsafe_allow_html=True,
        )

        # Parameters used
        params = status_data.get("parameters", {})
        if params:
            with st.expander("Execution Parameters"):
                for key, value in params.items():
                    st.text("{}: {}".format(key, value))

        # Step progress
        steps = status_data.get("steps", [])
        if steps:
            st.markdown("**Pipeline Steps:**")
            for step in steps:
                step_status = step.get("status", "Unknown")
                if step_status == "Succeeded":
                    icon = "[DONE]"
                elif step_status == "Executing":
                    icon = "[RUNNING]"
                elif step_status == "Failed":
                    icon = "[FAILED]"
                else:
                    icon = "[PENDING]"

                st.text("{} {} - {}".format(icon, step["name"], step_status))

                if step.get("failure_reason"):
                    st.error("Failure: {}".format(step["failure_reason"]))

        # Auto-refresh while executing
        if current_status in ("Executing", "Starting"):
            st.info("Pipeline is running. The page will refresh automatically.")
            time.sleep(15)
            st.rerun()

        # Show results when succeeded
        if current_status == "Succeeded":
            st.success("Pipeline completed successfully.")

            # Fetch artifacts
            if st.session_state.artifacts is None:
                with st.spinner("Loading results..."):
                    artifacts = api_request("GET", "/api/artifacts/{}".format(execution_arn))
                    if artifacts:
                        st.session_state.artifacts = artifacts

            if st.session_state.artifacts:
                artifacts = st.session_state.artifacts

                # -- EDA Plots --
                st.markdown("---")
                st.markdown("#### EDA Visualizations")

                eda_col1, eda_col2 = st.columns(2)

                with eda_col1:
                    heatmap_url = artifacts.get("correlation_heatmap_url", "")
                    if heatmap_url:
                        st.markdown("**Correlation Heatmap**")
                        st.image(heatmap_url, use_container_width=True)
                    else:
                        st.warning("Correlation heatmap not available.")

                with eda_col2:
                    missing_url = artifacts.get("missing_value_matrix_url", "")
                    if missing_url:
                        st.markdown("**Missing Value Matrix**")
                        st.image(missing_url, use_container_width=True)
                    else:
                        st.warning("Missing value matrix not available.")

                # -- Data Summary --
                data_summary = artifacts.get("data_summary", {})
                if data_summary:
                    with st.expander("Data Summary"):
                        ds_col1, ds_col2, ds_col3 = st.columns(3)
                        with ds_col1:
                            st.metric("Total Rows", data_summary.get("total_rows", "N/A"))
                        with ds_col2:
                            st.metric("Total Columns", data_summary.get("total_columns", "N/A"))
                        with ds_col3:
                            st.metric("Missing Values", data_summary.get("missing_values_total", "N/A"))

                # -- Evaluation Metrics --
                st.markdown("---")
                st.markdown("#### Model Performance Metrics")

                metrics = artifacts.get("evaluation_metrics", {})
                if metrics:
                    task = metrics.get("task_type", "")

                    if task == "classification":
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        with m_col1:
                            st.metric("Accuracy", "{:.4f}".format(metrics.get("accuracy", 0)))
                        with m_col2:
                            st.metric("Precision", "{:.4f}".format(metrics.get("precision", 0)))
                        with m_col3:
                            st.metric("Recall", "{:.4f}".format(metrics.get("recall", 0)))
                        with m_col4:
                            st.metric("F1 Score", "{:.4f}".format(metrics.get("f1_score", 0)))
                    elif task == "regression":
                        m_col1, m_col2, m_col3 = st.columns(3)
                        with m_col1:
                            st.metric("RMSE", "{:.4f}".format(metrics.get("rmse", 0)))
                        with m_col2:
                            st.metric("MAE", "{:.4f}".format(metrics.get("mae", 0)))
                        with m_col3:
                            st.metric("R2 Score", "{:.4f}".format(metrics.get("r2_score", 0)))

                    st.caption("Test samples: {}".format(metrics.get("test_samples", "N/A")))

                    # Show raw metrics JSON
                    with st.expander("Raw Metrics JSON"):
                        st.json(metrics)
                else:
                    st.warning("Evaluation metrics not available.")

                # -- Model Registry Info --
                model_arn = artifacts.get("model_package_arn", "")
                if model_arn:
                    st.markdown("---")
                    st.markdown("#### Model Registry")
                    st.text("Model Package ARN: {}".format(model_arn))
                    st.caption("Status: PendingManualApproval (approve in SageMaker Studio to deploy)")

        elif current_status == "Failed":
            st.error("Pipeline execution failed. Check the step details above for failure reasons.")

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Tabular ML Platform -- All modeling decisions are made by the user. "
    "No Auto-ML. The system executes your exact configuration."
)
