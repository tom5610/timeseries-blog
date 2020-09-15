
FROM python:3.7-slim-buster
    
COPY ./sql /opt/ml/processing/sql
    
RUN pip install pandas numpy geopandas scikit-learn fsspec s3fs boto3

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
