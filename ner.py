import torch
import numpy as np
import re

from functools import partial
from itertools import groupby
from operator import itemgetter
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from htx.base_model import TextTagger


class SpanLabeledTextDataset(Dataset):

    def __init__(
        self, list_of_strings, list_of_spans=None, tokenizer=None, tag_idx_map=None,
        cls_token='[CLS]', sep_token='[SEP]', pad_token_label_id=-1, max_seq_length=128, sep_token_extra=False,
        cls_token_at_end=False, sequence_a_segment_id=0, cls_token_segment_id=1, mask_padding_with_zero=True
    ):
        self.list_of_strings = list_of_strings
        self.list_of_spans = list_of_spans or [[] * len(list_of_strings)]
        self.tokenizer = tokenizer
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token_label_id = pad_token_label_id
        self.max_seq_length = max_seq_length
        self.sep_token_extra = sep_token_extra
        self.cls_token_at_end = cls_token_at_end
        self.sequence_a_segment_id = sequence_a_segment_id
        self.cls_token_segment_id = cls_token_segment_id
        self.mask_padding_with_zero = mask_padding_with_zero

        (self.original_list_of_tokens, self.original_list_of_tags, tag_idx_map_,
         original_list_of_tokens_start_map) = self._prepare_data()

        if tag_idx_map is None:
            self.tag_idx_map = tag_idx_map_
        else:
            self.tag_idx_map = tag_idx_map

        self.list_of_token_ids, self.list_of_label_ids, self.list_of_segment_ids, self.list_of_token_start_map = [], [], [], []

        for original_tokens, original_tags in zip(self.original_list_of_tokens, self.original_list_of_tags):
            token_ids, label_ids, segment_ids, token_start_map = self._convert_to_features(
                original_tokens, original_tags, self.tag_idx_map, original_list_of_tokens_start_map)
            self.list_of_token_ids.append(token_ids)
            self.list_of_segment_ids.append(segment_ids)
            self.list_of_label_ids.append(label_ids)
            self.list_of_token_start_map.append(token_start_map)

    def _convert_to_features(self, words, labels, label_map, list_token_start_map):
        tokens, label_ids, tokens_idx_map = [], [], []
        for i, (word, label, token_start_map) in enumerate(zip(words, labels, list_token_start_map)):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            tokens_idx_map.extend([token_start_map[i]] * len(word_tokens))
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [self.pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if self.sep_token_extra else 2
        if len(tokens) > self.max_seq_length - special_tokens_count:
            tokens = tokens[:(self.max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(self.max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [self.sep_token]
        label_ids += [self.pad_token_label_id]
        if self.sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [self.sep_token]
            label_ids += [self.pad_token_label_id]
        segment_ids = [self.sequence_a_segment_id] * len(tokens)
        if self.cls_token_at_end:
            tokens += [self.cls_token]
            label_ids += [self.pad_token_label_id]
            segment_ids += [self.cls_token_segment_id]
        else:
            tokens = [self.cls_token] + tokens
            label_ids = [self.pad_token_label_id] + label_ids
            segment_ids = [self.cls_token_segment_id] + segment_ids

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return token_ids, label_ids, segment_ids, tokens_idx_map

    def _apply_tokenizer(self, original_tokens, original_tags):
        out_tokens, out_tags, out_maps = [], [], []
        for i, (original_token, original_tag) in enumerate(zip(original_tokens, original_tags)):
            tokens = self.tokenizer.tokenize(original_token)
            out_tokens.extend(tokens)
            out_maps.extend([i] * len(tokens))
            start_tag = original_tag.startswith('B-')
            for j in range(len(tokens)):
                if (j == 0 and start_tag) or original_tag == 'O':
                    out_tags.append(original_tag)
                else:
                    out_tags.append(f'I-{original_tag[2:]}')
        return out_tokens, out_tags, out_maps

    def _prepare_data(self):
        list_of_tokens, list_of_tags, list_of_token_idx_maps = [], [], []
        tag_idx_map = {'O': 0}
        for text, spans in zip(self.list_of_strings, self.list_of_spans):
            if not text:
                continue

            tokens = []
            start = 0
            for t in text.split():
                tokens.append((t, start))
                start += len(t) + 1

            if spans:
                spans = list(sorted(spans, key=itemgetter('start')))
                span = spans.pop(0)
                prefix = 'B-'
                tags = []
                for token, token_start in tokens:
                    token_end = token_start + len(token) - 1
                    if not span or token_end < span['start']:
                        tags.append('O')
                    elif token_start > span['end']:
                        # this could happen if prev label ends with whitespaces, e.g. "cat " "too"
                        # TODO: it is not right choice to place empty tag here in case when current token is covered by next span  # noqa
                        tags.append('O')
                    else:
                        label = span['label']
                        if label.startswith(prefix):
                            tag = label
                        else:
                            tag = f'{prefix}{label}'
                        tags.append(tag)
                        if tag not in tag_idx_map:
                            tag_idx_map[tag] = len(tag_idx_map)
                        if span['end'] > token_end:
                            prefix = 'I-'
                        elif len(spans):
                            span = spans.pop(0)
                            prefix = 'B-'
                        else:
                            span = None
            else:
                tags = ['O'] * len(tokens)

            list_of_tokens.append([t[0] for t in tokens])
            list_of_token_idx_maps.append({i: t[1] for i, t in enumerate(tokens)})
            list_of_tags.append(tags)

        return list_of_tokens, list_of_tags, tag_idx_map, list_of_token_idx_maps

    def __len__(self):
        return len(self.list_of_token_ids)

    def __getitem__(self, idx):
        return {
            'tokens': self.list_of_token_ids[idx],
            'labels': self.list_of_label_ids[idx],
            'segments': self.list_of_segment_ids[idx],
            'token_start_map': self.list_of_token_start_map[idx],
        }

    @property
    def num_labels(self):
        return len(self.tag_idx_map)

    @classmethod
    def pad_sequences(cls, batch, mask_padding_with_zero, pad_on_left, pad_token, pad_token_segment_id, pad_token_label_id):
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        max_seq_length = max(map(len, (sample['tokens'] for sample in batch)))
        batch_input_ids, batch_label_ids, batch_segment_ids, batch_input_mask, batch_token_start_map = [], [], [], [], []
        batch_token_idx_map, batch_original_token_start_map = [], []
        for sample in batch:
            input_ids = sample['tokens']
            label_ids = sample['labels']
            segment_ids = sample['segments']
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += ([pad_token] * padding_length)
                input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids += ([pad_token_segment_id] * padding_length)
                label_ids += ([pad_token_label_id] * padding_length)
            batch_input_ids.append(input_ids)
            batch_label_ids.append(label_ids)
            batch_segment_ids.append(segment_ids)
            batch_input_mask.append(input_mask)
            batch_token_start_map.append(sample['token_start_map'])

        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'label_ids': torch.tensor(batch_label_ids, dtype=torch.long),
            'segment_ids': torch.tensor(batch_segment_ids, dtype=torch.long),
            'input_mask': torch.tensor(batch_input_mask, dtype=torch.long),
            'token_start_map': batch_token_start_map
        }


class TransformersBasedTagger(TextTagger):

    def load(self, train_output):
        pretrained_model = train_output['model_path']
        self._tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self._model = BertForTokenClassification.from_pretrained(pretrained_model)
        self._batch_size = train_output['batch_size']
        self._pad_token = self._tokenizer.convert_tokens_to_ids([self._tokenizer.pad_token])[0]
        self._pad_token_label_id = train_output['pad_token_label_id']
        self._label_map = train_output['label_map']
        self._mask_padding_with_zero = True

        self._batch_padding = partial(
            SpanLabeledTextDataset.pad_sequences,
            mask_padding_with_zero=self._mask_padding_with_zero,
            pad_on_left=False,
            pad_token=self._pad_token,
            pad_token_segment_id=0,
            pad_token_label_id=self._pad_token_label_id
        )

    def _make_span(self, tokens, tags):
        tokens = tokens[1:-1]
        tags = tags[1:-1]
        spans = []
        for label, it in groupby(zip(tokens, tags), key=lambda i: re.sub('^(B-|I-)', '', i[1])):
            toks, idx = list(zip(*(i[0] for i in it)))
            substr = ' '.join(toks)
            substr_start = idx[0]
            if label != 'O':
                spans.append({
                    'start': substr_start,
                    'end': substr_start + len(substr) - 1,
                    'label': label,
                    'substr': substr
                })
        return spans

    def predict(self, tasks):
        texts = list(map(lambda i: i['input'][0], tasks))
        predict_set = SpanLabeledTextDataset(texts, tokenizer=self._tokenizer)
        from_name = self.output_names[0]
        to_name = self.input_names[0]
        predict_loader = DataLoader(
            dataset=predict_set,
            batch_size=self._batch_size,
            collate_fn=self._batch_padding
        )

        mask_padding_token = 1 - int(self._mask_padding_with_zero)
        results = []
        for batch in tqdm(predict_loader, desc='Prediction'):
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['input_mask'],
                'token_type_ids': batch['segment_ids']
            }
            with torch.no_grad():
                model_output = self._model(**inputs)
                logits = model_output[0]

            batch_preds = logits.detach().cpu().numpy()
            argmax_batch_preds = np.argmax(batch_preds, axis=-1)
            max_batch_preds = np.max(batch_preds, axis=-1)
            input_mask = batch['input_mask'].detach().cpu().numpy()
            batch_token_start_map = batch['token_start_map']

            for max_preds, argmax_preds, mask_tokens, token_start_map in zip(
                max_batch_preds, argmax_batch_preds, input_mask, batch_token_start_map
            ):
                preds, scores, starts = [], [], []
                for max_pred, argmax_pred, mask_token, token_start in zip(max_preds, argmax_preds, mask_tokens, token_start_map):
                    if mask_token != mask_padding_token:
                        preds.append(self._label_map[argmax_pred])
                        scores.append(max_pred)
                        starts.append(token_start)
                mean_score = np.mean(scores) if len(scores) > 0 else 0

                result = []

                for label, group in groupby(zip(preds, starts, scores), key=lambda i: re.sub('^(B-|I-)', '', i[0])):
                    _, group_starts, group_scores = group
                    group_start = group_starts[0]
                    if len(result) > 0:
                        result[-1]['value']['end'] = group_start - 1
                    if label != 'O':
                        result.append({
                            'from_name': from_name,
                            'to_name': to_name,
                            'type': 'labels',
                            'value': {
                                'labels': [label],
                                'start': group_starts[0],
                                'end': None
                            }
                        })
                results.append({
                    'result': result,
                    'score': mean_score,
                    'cluster': None
                })
        return results


def train_ner(
    input_data, output_model_dir, pretrained_model,
    batch_size=32, learning_rate=5e-5, adam_epsilon=1e-8, num_train_epochs=3, weight_decay=0.0, logging_steps=10,
    warmup_steps=0, save_steps=50,
    **kwargs):

    # read input data stream
    texts, list_of_spans = [], []
    for item in input_data:
        texts.append(item['input'][0])
        list_of_spans.append(item['output'])

    pad_token_label_id = CrossEntropyLoss().ignore_index
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    train_set = SpanLabeledTextDataset(texts, list_of_spans, tokenizer, pad_token_label_id=pad_token_label_id)

    batch_padding = partial(
        SpanLabeledTextDataset.pad_sequences,
        mask_padding_with_zero=True,
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
        pad_token_label_id=pad_token_label_id
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=batch_padding
    )
    config = BertConfig.from_pretrained(pretrained_model)
    config.num_labels = train_set.num_labels

    model = BertForTokenClassification.from_pretrained(pretrained_model, config=config)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    num_training_steps = len(train_loader) * num_train_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    tr_loss, logging_loss = 0, 0
    global_step = 0
    tb_writer = SummaryWriter()
    epoch_iterator = trange(num_train_epochs, desc='Epoch')
    for _ in epoch_iterator:
        batch_iterator = tqdm(train_loader, desc='Batch')
        for step, batch in enumerate(batch_iterator):

            model.train()
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['input_mask'],
                'labels': batch['label_ids'],
                'token_type_ids': batch['segment_ids']
            }
            model_output = model(**inputs)
            loss = model_output[0]
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            if global_step % logging_steps == 0:
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss) / logging_steps, global_step)
                logging_loss = tr_loss

    tb_writer.close()

    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    label_map = {i: t for t, i in train_set.tag_idx_map.items()}

    return {
        'model_path': output_model_dir,
        'batch_size': batch_size,
        'pad_token_label_id': pad_token_label_id,
        'label_map': label_map
    }
