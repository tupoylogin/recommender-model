FROM python:3
ADD app/ /app
WORKDIR /app
RUN apt-get update
RUN pip install -r requirements.txt
RUN pip freeze > requirements.txt
CMD python app.py