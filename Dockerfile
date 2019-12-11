FROM python:3.7
RUN pip install uwsgi supervisor
WORKDIR /app
COPY requirements.txt /app/
EXPOSE 9090
RUN pip install -r requirements.txt

ADD . /app
CMD ["supervisord", "-c", "supervisord.conf"]