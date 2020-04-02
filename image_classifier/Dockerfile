FROM python:3.7
RUN pip install uwsgi
RUN pip install supervisor
EXPOSE 9090
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt

ADD . /app
CMD ["supervisord", "-c", "supervisord.conf"]