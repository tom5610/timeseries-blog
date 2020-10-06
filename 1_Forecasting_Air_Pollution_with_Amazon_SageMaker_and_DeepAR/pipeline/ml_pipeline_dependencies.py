import uuid
import time
import boto3
import argparse
import os, urllib.request
import stepfunctions
from stepfunctions.inputs import ExecutionInput
from stepfunctions.steps.sagemaker import *
from stepfunctions.steps.states import *
from stepfunctions.workflow import Workflow
from stepfunctions.steps import *

import sagemaker
from sagemaker import get_execution_role
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
from sagemaker.model import Model

session = boto3.Session()
region = session.region_name
account_id = session.client('sts').get_caller_identity().get('Account')
bucket_name = f'openaq-forecasting-{account_id}-{region}'

sagemaker_session = sagemaker.Session()
role = get_execution_role()

S3_KEY_TRAINED_MODEL = "sagemaker/model/model.tar.gz"
EXISTING_MODEL_URI = f"s3://{bucket_name}/{S3_KEY_TRAINED_MODEL}"

def setup_trained_model(bucket_name, s3_key_trained_model):
    # upload existing model artifact to working bucket
    s3 = boto3.client('s3')

    os.makedirs('model', exist_ok=True)
    urllib.request.urlretrieve('https://d8pl0xx4oqh22.cloudfront.net/model.tar.gz', 'model/model.tar.gz')
    s3.upload_file('model/model.tar.gz', bucket_name, s3_key_trained_model)
    
if __name__ == "__main__":
    setup_trained_model(bucket_name, S3_KEY_TRAINED_MODEL)