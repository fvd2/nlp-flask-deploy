FROM python:3.8.5

WORKDIR /app

RUN pip install scikit-learn flask gunicorn

ADD ./models ./models
ADD server.py server.py

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]
