FROM python:3.8-slim-buster

RUN pip3 install pandas==1.1.4 numpy==1.19.4 scikit-learn==0.23.2 scipy==1.5.4 boto3==1.17.12

WORKDIR /home

COPY src/* /home/

ENTRYPOINT ["python3", "drift_detector.py"]