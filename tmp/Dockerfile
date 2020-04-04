FROM python:3.7
RUN pip install uwsgi supervisor
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt

ADD .. /app
EXPOSE 80
CMD ["supervisord", "-c", "supervisord.conf"]