import sagemaker
from boto3 import session
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import EstimatorBase
from sagemaker import image_uris

import time
import os
# Initialize the SageMaker session

# image_uris.retrieve(framework='pytorch',region='us-east-1',version='1.8.0',py_version='py3',image_scope='inference', instance_type='ml.c5.4xlarge')

def fit_with_retries(retries: int, estimator: EstimatorBase, *args, **kwargs):
    """Run estimator fit with retries in case of temporary issues like capacity exception or user exceeded resource usage
    Example invocation: fit_with_retries(5, estimator, job_name="my job name")
    Args:
        retries (int): How many retries in case of exception_to_try is raised
        estimator (EstimatorBase): will call estimator.fit(...)
        *args: list of positioned arguments to pass to fit()
        **kwargs: list of keyword arguments to pass to fit()
    Returns:
        None
    """
    orig_job_name = (
        kwargs["job_name"]
        if "job_name" in kwargs and kwargs["job_name"]
        else None
    )
    for i in range(1, retries + 1):
        try:
            # Ensure job_name is unique between retries (if specified)
            if orig_job_name:
                kwargs["job_name"] = orig_job_name + f"-{i}"
            estimator.fit(*args, **kwargs)
            break
        except Exception as e:
            if not (
                "CapacityError" in str(e) or "ResourceLimitExceeded" in str(e)
            ):
                raise e
            print(f"Caught error: {type(e).__name__}: {e}")
            if i == retries:
                print(f"Giving up after {retries} failed attempts.")
                raise e
            else:
                if "ResourceLimitExceeded" in str(e):
                    seconds = 10
                    print(
                        f"ResourceLimitExceeded: Sleeping {seconds}s before retrying."
                    )
                    time.sleep(seconds)
                print(f"Retrying attempt: {i+1}/{retries}")
                continue

# region_name = "us-west-2"
region_name = "us-east-1"

# boto_session = session.Session(region_name="us-west-2")  # us-west-2
boto_session = session.Session(region_name=region_name)  # us-west-2
sm_session = sagemaker.Session(boto_session=boto_session)

# Define the IAM role with necessary permissions
role = "arn:aws:iam::542375318953:role/accelerate_sagemaker_execution_role"

# Define the S3 bucket and prefix for input and output data
bucket = 'eiga-training'
prefix = 'davidi/pusle-testing'

instance_type = "ml.p4de.24xlarge"
# instance_type = "ml.g6e.2xlarge"

# Define the input data location
# train_input = TrainingInput(
#     s3_data=f's3://{bucket}/{prefix}/input',
#     content_type='application/x-image'
# )

environment = {
    # "ACCELERATE_USE_SAGEMAKER": "true", #disabled for FSDP
    "ACCELERATE_MIXED_PRECISION": "bf16",
    "ACCELERATE_DYNAMO_BACKEND": "NO",
    "ACCELERATE_DYNAMO_MODE": "default",
    "ACCELERATE_DYNAMO_USE_FULLGRAPH": "False",
    "ACCELERATE_DYNAMO_USE_DYNAMIC": "False",
    #Before FSDP
    # "ACCELERATE_SAGEMAKER_DISTRIBUTED_TYPE": "DATA_PARALLEL",
    #After FSDP
    # "ACCELERATE_USE_FSDP": "true", #- leave this blank ?
    "CLOUD_PROVIDER":'AWS',
    # NCCL STUFF
    # "NCCL_DEBUG":"INFO",
    # "NCCL_DEBUG_SUBSYS":"COLL",
}

job_name=f'test-pulse-{time.strftime("%Y-%m-%d-%H-%M-%S")}'
print(f"Job name: {job_name}")
# Create a PyTorch Estimator
estimator = PyTorch(
    job_name=job_name,
    entry_point='examples/bria4B_adapt/example_train.py',
    # entry_point='examples/bria4B_adapt/example_train.py',
    source_dir='/home/ubuntu/gradient/',
    instance_type=instance_type,
    role=role,
    use_spot_instances=False,
    tags=[{"Key": "billing_category","Value": "project-pulse"}],
    instance_count=1,
    image_uri="542375318953.dkr.ecr.us-east-1.amazonaws.com/pulse/bria-4b-adapt:pip",
    distribution={"torch_distributed": {"enabled": True}},
    output_path=f's3://{bucket}/{prefix}/output',
    sagemaker_session=sm_session,
    max_run=432000,
    # max_wait=1200*2,
    environment=environment,
    metric_definitions=None,
    debugger_hook_config=False,
)
fit_with_retries(10, estimator,job_name=job_name)

