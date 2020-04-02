import os
import logging
import argparse

logging.basicConfig(level=logging.INFO)

from htx import app, init_model_server
from image_classifier.image_classifier import ResnetSklearnImageClassifier

init_model_server(
    create_model_func=ResnetSklearnImageClassifier,
    train_script=ResnetSklearnImageClassifier.fit,
    image_folder='~/.heartex/images',
    redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
    redis_host=os.environ.get('REDIS_HOST', 'localhost')
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', dest='port', default='10001')
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=True)
