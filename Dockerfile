FROM python:3.8.5

WORKDIR /app

RUN pip install --upgrade pip 
RUN pip install pandas scikit-learn flask gunicorn

ADD ./model ./model
ADD server.py server.py

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]
