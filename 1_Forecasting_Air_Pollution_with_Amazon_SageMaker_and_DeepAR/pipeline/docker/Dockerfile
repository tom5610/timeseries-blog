
FROM python:3.7-slim-buster
    
COPY ./sql /opt/ml/processing/
    
RUN pip install PyAthena[Pandas] geopandas scikit-learn
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
