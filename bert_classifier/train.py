import torch
import os
import numpy as np

from tqdm import tqdm, trange
from collections import deque
from tensorboardX import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.preprocessing.label import LabelEncoder


if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def pad_sequences(input_ids, maxlen):
    padded_ids = []
    for ids in input_ids:
        nonpad = min(len(ids), maxlen)
        pids = [ids[i] for i in range(nonpad)]
        for i in range(nonpad, maxlen):
            pids.append(0)
        padded_ids.append(pids)
    return padded_ids


def prepare_texts(texts, tokenizer, maxlen, sampler_class, batch_size, choices_ids=None):
    # create input token indices
    input_ids = []
    for text in texts:
        input_ids.append(tokenizer.encode(text, add_special_tokens=True))
    # input_ids = pad_sequences(input_ids, maxlen=maxlen, dtype='long', value=0, truncating='post', padding='post')
    input_ids = pad_sequences(input_ids, maxlen)
    # Create attention masks
    attention_masks = []
    for sent in input_ids:
        attention_masks.append([int(token_id > 0) for token_id in sent])

    if choices_ids is not None:
        dataset = TensorDataset(torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long), torch.tensor(choices_ids, dtype=torch.long))
    else:
        dataset = TensorDataset(torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long))
    sampler = sampler_class(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def calc_slope(y):
    n = len(y)
    if n == 1:
        raise ValueError('Can\'t compute slope for array of length=1')
    x_mean = (n + 1) / 2
    x2_mean = (n + 1) * (2 * n + 1) / 6
    xy_mean = np.average(y, weights=np.arange(1, n + 1))
    y_mean = np.mean(y)
    slope = (xy_mean - x_mean * y_mean) / (x2_mean - x_mean * x_mean)
    return slope


def train_classifier(
    input_data, output_dir, pretrained_model='bert-base-multilingual-cased',
    cache_dir=None, maxlen=64, batch_size=32, num_epochs=100, logging_steps=1,
    train_logs=None, **kwargs
):

    # read input data stream
    texts, choices = [], []
    for item in input_data:
        texts.append(item['input'][0])
        choices.append(item['output'][0])

    le = LabelEncoder()
    choices_ids = le.fit_transform(choices)

    tokenizer = BertTokenizer.from_pretrained(pretrained_model, cache_dir=cache_dir)

    train_dataloader = prepare_texts(texts, tokenizer, maxlen, RandomSampler, batch_size, choices_ids)

    model = BertForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=len(le.classes_),
        output_attentions=False,
        output_hidden_states=False,
        cache_dir=cache_dir
    )
    model.to(device)

    total_steps = len(train_dataloader) * num_epochs
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    global_step = 0
    total_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(num_epochs, desc='Epoch')
    if train_logs:
        tb_writer = SummaryWriter(logdir=os.path.join(train_logs, os.path.basename(output_dir)))
    else:
        tb_writer = None
    loss_queue = deque(maxlen=10)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[2]}
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            if global_step % logging_steps == 0:
                last_loss = (total_loss - logging_loss) / logging_steps
                loss_queue.append(last_loss)
                if tb_writer:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', last_loss, global_step)
                logging_loss = total_loss

        # slope-based early stopping
        if len(loss_queue) == loss_queue.maxlen:
            slope = calc_slope(loss_queue)
            if tb_writer:
                tb_writer.add_scalar('slope', slope, global_step)
            if abs(slope) < 1e-2:
                break

    if tb_writer:
        tb_writer.close()

    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        'model_path': output_dir,
        'batch_size': batch_size,
        'maxlen': maxlen,
        'pretrained_model': pretrained_model,
        'choices_map': list(map(str, le.classes_))
    }
