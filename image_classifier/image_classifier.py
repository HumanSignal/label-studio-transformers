import logging
import os
import torch
import numpy as np
import json
import pickle
import requests
import io
import hashlib

from htx.base_model import SingleClassImageClassifier
from htx.utils import encode_labels
from requests.auth import HTTPBasicAuth

from sklearn.linear_model import LogisticRegression
from torch import nn, no_grad
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEFAULT_HOSTNAME = 'http://localhost:8200'

image_size = 224
preprocessing = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
resnet = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
resnet.eval()


class ImageClassifierDataset(Dataset):

    def __init__(self, inputs, outputs, image_folder, login=None, password=None):
        self.image_folder = image_folder
        self.inputs = []
        self.outputs = []
        for image, output in zip(inputs, outputs):
            try:
                self.inputs.append(self.prepare_image(image, self.image_folder, login, password))
            except Exception as exc:
                logger.warning(f'Error while processing image {image}. Reason: {exc}', exc_info=True)
                pass
            else:
                self.outputs.append(output)

    @classmethod
    def _download_file(cls, url, filepath, login=None, password=None):
        logger.info(f'Downloading {url} to {filepath}')
        try:
            if url.startswith('/'):
                url = DEFAULT_HOSTNAME + url
            if login and password:
                r = requests.get(url, auth=HTTPBasicAuth(login, password))
            else:
                r = requests.get(url)
            r.raise_for_status()
            with io.open(filepath, mode='wb') as fout:
                fout.write(r.content)
        except:
            logger.error(f'Failed download {url} to {filepath}', exc_info=True)
            return None
        else:
            return filepath

    @classmethod
    def _get_image_from_url(self, url):
        logger.info(f'Getting image from {url}')
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with io.BytesIO(r.content) as f:
            return Image.open(f).convert('RGB')

    @classmethod
    def prepare_image_no_store(cls, image_url):
        image_data = cls._get_image_from_url(image_url)
        preprocessed_image_data = preprocessing(image_data)
        return preprocessed_image_data

    @classmethod
    def prepare_image(cls, image_url, image_folder, login=None, password=None):
        filename = hashlib.md5(image_url.encode()).hexdigest()
        filepath = os.path.join(image_folder, filename)
        if not os.path.exists(filepath):
            filepath = cls._download_file(image_url, filepath, login, password)
        else:
            logger.info(f'File {filepath} already exists.')
        try:
            image_data = Image.open(filepath).convert('RGB')
        except Exception as e:
            logger.error(str(filepath) + ' :: ' + str(e))
            return None
        logger.info('Step 2')
        preprocessed_image_data = preprocessing(image_data)
        logger.info('Step 3')
        return preprocessed_image_data

    def __getitem__(self, index):
        #image = self.prepare_image(self.inputs[index], self.image_folder)
        image = self.inputs[index]
        image_class = self.outputs[index]
        return image, image_class

    def __len__(self):
        return len(self.inputs)


class ImageClassifierModel(object):

    def __init__(self, image_folder):
        self.image_folder = os.path.expanduser(image_folder)
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        self._model = None

    def fit(self, inputs, outputs, login=None, password=None):
        dataset = ImageClassifierDataset(inputs, outputs, self.image_folder, login, password)
        dataloader = DataLoader(dataset, batch_size=128, num_workers=0)

        X, y = [], []
        with no_grad():
            for batch_inputs, batch_outputs in dataloader:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                batch_X = resnet(batch_inputs).to(device)
                batch_X = torch.reshape(batch_X, (batch_X.size(0), batch_X.size(1)))
                X.append(batch_X.data.cpu().numpy())
                y.append(batch_outputs.data.cpu().numpy())

        X = np.vstack(X)
        y = np.hstack(y)

        self._model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        self._model.fit(X, y)
        logger.info(f'Model {self.__class__.__name__} fit finished.')
        return True

    def predict_proba(self, inputs, login=None, password=None):
        preprocessed_images = []
        for image_url in inputs:
            preprocessed_images.append(
                ImageClassifierDataset.prepare_image(image_url, self.image_folder, login, password))
        logger.info('Step X1')
        preprocessed_images = torch.stack(preprocessed_images)
        logger.info('Step X2')
        with no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor_X = resnet(preprocessed_images).to(device)
            logger.info('Step X3')
            tensor_X = torch.reshape(tensor_X, (tensor_X.size(0), tensor_X.size(1)))
            logger.info('Step X4')
            X = tensor_X.cpu().data.numpy()
            logger.info('Step X5')
        return self._model.predict_proba(X)


class ResnetSklearnImageClassifier(SingleClassImageClassifier):

    def load(self, serialized_train_output):
        train_output = json.loads(serialized_train_output)
        model_dir = train_output['model_dir']
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f'Can\'t load from {model_dir}: directory doesn\' exist.')

        # load model
        model_file = os.path.join(model_dir, 'model.pkl')
        if not os.path.exists(model_file):
            raise FileNotFoundError(f'Can\'t load model from {model_file}: file doesn\' exist.')
        with open(model_file, mode='rb') as f:
            self._model = pickle.load(f)

        # load idx2label
        labels_file = os.path.join(model_dir, 'idx2label.json')
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f'Can\'t load label indices from {labels_file}: file doesn\' exist.')

        with open(labels_file) as f:
            self._idx2label = json.load(f)

    def predict(self, tasks, login=None, password=None, **kwargs):
        image_urls = []
        for item in tasks:
            if len(item['input']) < 1:
                logger.error(f'Item {item} is invalid.')
                continue
            image_urls.append(item['input'][0])

        # convert probabilities array into heartex predictions
        predict_proba = self._model.predict_proba(image_urls, login, password)
        predict_idx = np.argmax(predict_proba, axis=1)
        predict_scores = predict_proba[np.arange(len(predict_idx)), predict_idx]
        predict_labels = [self._idx2label[c] for c in predict_idx]
        return self.make_results(tasks, predict_labels, predict_scores)

    @classmethod
    def fit(cls, input_data, output_model_dir, image_folder, login=None, password=None, **kwargs):
        logging.basicConfig(level=logging.INFO)

        # collect data
        input_images, output_labels = [], []
        for item in input_data:
            if len(item['input']) < 1 or not item['output']:
                logger.error(f'Item {item} is invalid.')
                continue
            input_images.append(item['input'][0])
            output_labels.append(item['output'][0])

        # create label indexers
        idx2label, output_labels_idx = encode_labels(output_labels)
        if len(idx2label) < 2:
            raise ValueError(f'Unable to start training with less than two classes: {idx2label}')

        model = ImageClassifierModel(image_folder)
        model.fit(input_images, output_labels_idx, login, password)

        labels_file = os.path.join(output_model_dir, 'idx2label.json')
        with open(labels_file, mode='w') as fout:
            json.dump(idx2label, fout, indent=4)

        model_file = os.path.join(output_model_dir, 'model.pkl')
        with open(model_file, mode='wb') as fout:
            pickle.dump(model, fout)

        logger.info(f'Model training finished, created new resources in {output_model_dir}')
        return json.dumps({'model_dir': output_model_dir})
