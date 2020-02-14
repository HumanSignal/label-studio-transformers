import os

from htx import app, init_model_server
from bert_classifier.train import train_classifier
from bert_classifier.serve import LabelStudioTransformersClassifier


init_model_server(
    create_model_func=LabelStudioTransformersClassifier,
    train_script=train_classifier,
    # name of the Redis queue
    redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
    # Redis host
    redis_host=os.environ.get('REDIS_HOST', 'localhost'),
    # here we pass the kwargs parameters to train script
    pretrained_model=os.environ.get('pretrained_model', 'bert-base-multilingual-cased'),
    cache_dir=os.environ.get('cache_dir', '/data/cache'),
    model_dir=os.environ.get('model_dir', '/data/model'),
    train_logs=os.environ.get('train_logs', '/data/train_logs')
)
