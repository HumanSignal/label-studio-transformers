import torch
import numpy as np

from htx.base_model import SingleClassTextClassifier

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import SequentialSampler

from bert_classifier.train import prepare_texts


device = 'cpu'


class LabelStudioTransformersClassifier(SingleClassTextClassifier):

    def load(self, train_output):
        pretrained_model = train_output['model_path']
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.model = BertForSequenceClassification.from_pretrained(pretrained_model)
        self.model.to(device)
        self.model.eval()
        self.batch_size = train_output['batch_size']
        self.choices_map = train_output['choices_map']
        self.maxlen = train_output['maxlen']

    def predict(self, tasks, **kwargs):
        texts = list(map(lambda i: i['input'][0], tasks))
        predict_dataloader = prepare_texts(texts, self.tokenizer, self.maxlen, SequentialSampler, self.batch_size)

        pred_labels, pred_scores = [], []
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1]
            }
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs[0]

            batch_preds = logits.detach().cpu().numpy()

            argmax_batch_preds = np.argmax(batch_preds, axis=-1)
            pred_labels.extend(str(self.choices_map[i]) for i in argmax_batch_preds)

            max_batch_preds = np.max(batch_preds, axis=-1)
            pred_scores.extend(float(s) for s in max_batch_preds)

        return self.make_results(tasks, pred_labels, pred_scores)
