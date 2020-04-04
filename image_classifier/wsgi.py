import os
import logging
import argparse

logging.basicConfig(level=logging.INFO)

from htx import app, init_model_server
from image_classifier import ResnetSklearnImageClassifier

init_model_server(
    create_model_func=ResnetSklearnImageClassifier,
    train_script=ResnetSklearnImageClassifier.fit,
    image_folder='~/.heartex/images',
    redis_queue=os.environ.get('RQ_QUEUE_NAME', 'label-studio'),
    redis_host=os.environ.get('REDIS_HOST', 'localhost'),
    cache_dir=os.environ.get('cache_dir', 'storage/image_classifier/cache'),
    model_dir=os.environ.get('model_dir', 'storage/image_classifier/model'),
    train_logs=os.environ.get('train_logs', 'storage/image_classifier/train_logs')
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', dest='port', default='9090')
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=False)
