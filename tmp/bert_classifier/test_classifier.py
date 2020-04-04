import os
import pandas as pd
import json
import io

from train import train_classifier
from serve import LabelStudioTransformersClassifier
from sklearn.metrics.classification import accuracy_score

test_dir = 'tmp_test_dir'


def test_train_classifier():

    data = pd.read_csv('test_data/amazon_cells_labelled.tsv', sep='\t')
    data = data.sample(frac=1)

    train_data = []
    n_train = 200
    for row in data.iloc[:n_train].itertuples(index=False):
        train_data.append({'input': [row.text], 'output': [row.sentiment]})

    test_data = []
    for row in data.iloc[n_train:].itertuples(index=False):
        test_data.append({'input': [row.text], 'output': [row.sentiment]})

    output_dir = test_dir
    os.makedirs(output_dir, exist_ok=True)
    resources = train_classifier(train_data, output_dir, num_epochs=5)
    with io.open(os.path.join(output_dir, 'resources.json'), mode='w') as fout:
        json.dump(resources, fout, indent=2)

    with io.open(os.path.join(output_dir, 'test_dataset.json'), mode='w') as fout:
        json.dump(test_data, fout, indent=2)


def test_serve_classifier():
    with io.open(os.path.join(test_dir, 'test_dataset.json')) as f:
        test_data = json.load(f)

    with io.open(os.path.join(test_dir, 'resources.json')) as f:
        resources = json.load(f)

    model = LabelStudioTransformersClassifier(input_names=['text'], output_names=['label'])
    model.load(resources)

    true_choices = []
    pred_choices = []
    for task in test_data:
        task['id'] = 0
        true_choice = str(task['output'][0])
        task['output'][0] = true_choice
        output = model.predict([task])
        pred_choice = output[0]['result'][0]['value']['choices'][0]
        print(true_choice, '=>', pred_choice)
        true_choices.append(true_choice)
        pred_choices.append(pred_choice)

    print(accuracy_score(true_choices, pred_choices))


if __name__ == '__main__':
    test_train_classifier()
    test_serve_classifier()
