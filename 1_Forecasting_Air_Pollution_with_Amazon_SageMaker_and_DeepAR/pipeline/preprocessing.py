
import argparse
import os
import warnings

import boto3, time, s3fs, json, warnings, os
import urllib.request
from datetime import date, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from multiprocessing import Pool

# the train test split date is used to split each time series into train and test sets
train_test_split_date = date.today() - timedelta(days = 30)

# the sampling frequency determines the number of hours per sample
# and is used for aggregating and filling missing values
frequency = '1'

# prediction length is how many hours into future to predict values for
prediction_length = 48

# context length is how many prior time steps the predictor needs to make a prediction
context_length = 3

warnings.filterwarning('ignore')

session = boto3.Session()
sagemaker_session = sagemaker.Session()
region = session.region_name
account = session.client('sts').get_caller_identity().get('Account')
bucket_name = f"{account_id}-openaq-lab"
athena_s3_staging_dir = f's3://{bucket_name}/athena/results/'

s3 = boto3.client('s3')

# @todo to evaluate whether we should store existing model.tar.gz onto s3 bucket.

# processing Athena
def athena_query_table(query_file, wait=None):
    results_uri = athena_execute(query_file, 'csv', wait)
    return results_uri

def athena_execute(query_file, ext, wait):
    with open(query_file) as f:
        query_str = f.read()  
        
    athena = boto3.client('athena')
    s3_dest = athena_s3_staging_dir
    query_id = athena.start_query_execution(
        QueryString= query_str, 
         ResultConfiguration={'OutputLocation': athena_s3_staging_dir}
    )['QueryExecutionId']
        
    results_uri = f'{athena_s3_staging_dir}{query_id}.{ext}'
        
    start = time.time()
    while wait == None or wait == 0 or time.time() - start < wait:
        result = athena.get_query_execution(QueryExecutionId=query_id)
        status = result['QueryExecution']['Status']['State']
        if wait == 0 or status == 'SUCCEEDED':
            break
        elif status in ['QUEUED','RUNNING']:
            continue
        else:
            raise Exception(f'query {query_id} failed with status {status}')

            time.sleep(3) 

    return results_uri       

def get_sydney_openaq_data(sql_query_file_path = "/opt/ml/processing/sql/sydney.dml"):
    query_results_uri = athena_query_table(sql_query_file_path)
    print (f'reading {query_results_uri}')
    raw = pd.read_csv(query_results_uri, parse_dates=['timestamp'])
    return raw

def fill_missing_hours(df):
    df = df.reset_index(level=categorical_levels, drop=True)                                    
    index = pd.date_range(df.index.min(), df.index.max(), freq='1H')
    return df.reindex(pd.Index(index, name='timestamp'))

def featurize(raw):
    # Sort and index by location and time
    categorical_levels = ['country', 'city', 'location', 'parameter']
    index_levels = categorical_levels + ['timestamp']
    indexed = raw.sort_values(index_levels, ascending=True)
    indexed = indexed.set_index(index_levels)
    # indexed.head()    
    
    # Downsample to hourly samples by maximum value
    downsampled = indexed.groupby(categorical_levels + [pd.Grouper(level='timestamp', freq='1H')]).max()

    # Back fill missing values
    filled = downsampled.groupby(level=categorical_levels).apply(fill_missing_hours)
    filled[filled['value'].isnull()].groupby('location').count().describe()
    
    filled['value'] = filled['value'].interpolate().round(2)
    filled['point_latitude'] = filled['point_latitude'].fillna(method='pad')
    filled['point_longitude'] = filled['point_longitude'].fillna(method='pad')

    # Create Features
    aggregated = filled.reset_index(level=4)\
        .groupby(level=categorical_levels)\
        .agg(dict(timestamp='first', value=list, point_latitude='first', point_longitude='first'))\
        .rename(columns=dict(timestamp='start', value='target'))    
    aggregated['id'] = np.arange(len(aggregated))
    aggregated.reset_index(inplace=True)
    aggregated.set_index(['id']+categorical_levels, inplace=True)
    
    metadata = gpd.GeoDataFrame(
        aggregated.drop(columns=['target','start']), 
        geometry=gpd.points_from_xy(aggregated.point_longitude, aggregated.point_latitude), 
        crs={"init":"EPSG:4326"}
    )
    metadata.drop(columns=['point_longitude', 'point_latitude'], inplace=True)
    # set geometry index
    metadata.set_geometry('geometry')

    # Add Categorical features
    level_ids = [level+'_id' for level in categorical_levels]
    for l in level_ids:
        aggregated[l], index = pd.factorize(aggregated.index.get_level_values(l[:-3]))

    aggregated['cat'] = aggregated.apply(lambda columns: [columns[l] for l in level_ids], axis=1)
    features = aggregated.drop(columns=level_ids+ ['point_longitude', 'point_latitude'])
    features.reset_index(level=categorical_levels, inplace=True, drop=True)
    
    return features

def split_train_test_data(features, days = 30):
    train_test_split_date = date.today() - timedelta(days = days)
    train = filter_dates(features, None, train_test_split_date, '1H')
    test = filter_dates(features, train_test_split_date, None, '1H')
    return train, test

raw = get_sydney_openaq_data()
features = featurize(raw)
train, test = split_train_test_data(features)

# upload dataset to S3.
local_data_path = '/opt/ml/processing/data'
os.makedirs(local_data_path, exist_ok = True)
train.to_json(f'{local_data_path}/train.json', orient='records', lines = True)
train_data_uri = sagemaker_session.upload_data(path=f'{local_data_path}/train.json', key_prefix = 'preprocessing_data')

test.to_json(f'{local_data_path}/test.json', orient='records', lines = True) 
test_data_uri =  sagemaker_session.upload_data(path=f'{local_data_path}/test.json', key_prefix = 'preprocessing_data')

features.to_json(f'{local_data_path}/all_data.json', orient='records', lines = True)
all_data_uri = sagemaker_session.upload_data(path=f'{local_data_path}/all_data.json', key_prefix = 'preprocessing_data')
