FROM python:3.8

RUN pip install --upgrade pip==20.3.1

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /duke-dbt
