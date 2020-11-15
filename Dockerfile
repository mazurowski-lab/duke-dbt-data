FROM python:3.8

RUN pip install --upgrade pip==20.2.4

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /duke-dbt
