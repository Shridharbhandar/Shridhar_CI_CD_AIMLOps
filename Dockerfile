FROM python:3.7

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

RUN python train.py

CMD ["python", "test.py"]