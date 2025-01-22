import os
import sagemaker
from sagemaker.pytorch import PyTorch

# SageMaker session
sagemaker_session = sagemaker.Session()

# Role ARN
role = "arn:aws:iam::542375318953:role/accelerate_sagemaker_execution_role"
fs_id = "fs-090516e1ff1212ff2"
fs_mount = "2dpzjbev"

# Training job configuration
torch_estimator = PyTorch(
    entry_point="examples/bria4B_adapt/example_train.py",  # Replace with your script name
    source_dir=os.path.dirname(os.path.realpath(__file__)),  # Current directory
    instance_type="ml.p4de.24xlarge",  # Adjust instance type as needed
    instance_count=1,  # Number of instances
    role=role,
    framework_version="2.0.0",
    py_version="py39",
    hyperparameters={},  # Pass if needed
    debugger_hook_config=False,
    distribution={
        "torch_distributed": {"enabled": True}
    },  # Enable distributed training
    checkpoint_s3_uri="s3://your-bucket/checkpoints",  # S3 path for checkpoints
    checkpoint_local_path="/opt/ml/checkpoints",
    input_mode="FastFile",  # FastFile for efficient S3 access
    use_spot_instances=True,
    max_run=432000,  # 5 days
    sagemaker_session=sagemaker_session,
)

# Launch the training job
torch_estimator.fit({"training": "s3://your-bucket/training-data"})
