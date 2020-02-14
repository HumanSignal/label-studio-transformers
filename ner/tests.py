import os

from ner import SpanLabeledTextDataset
from transformers import BertTokenizer


def data_for_test_1():
    spans = [{
        "end": 169,
        "label": "Task",
        "start": 75,
        "text": " https://github.com/heartexlabs/label-studio/pull/157 \u0442\u0443\u0442 \u0431\u044b\u043b \u0435\u0449\u0435 \u043e\u0434\u0438\u043d \u043a\u043e\u043c\u043c\u0438\u0442 \u043a\u0443\u0434\u0430 \u043e\u043d \u043f\u0440\u043e\u0441\u0440\u0430\u043b\u0441\u044f"
    }, {
        "end": 282,
        "label": "Task",
        "start": 248,
        "text": "\u043e\u0448\u0438\u0431\u043a\u0430 \u0432\u043e\u0437\u043d\u0438\u043a\u0430\u0435\u0442 LSB is not defined"
    }, {
        "end": 57,
        "label": "EmotionDOWN",
        "start": 54,
        "text": "\u0431\u043b\u0438\u043d"
    }]
    text = u"\u0430, \u043e\u043a\nnik \u00a02:27 PM MSK\u00a0uploaded this image: image.png\n\u0431\u043b\u0438\u043d \u043d\u0435 \u043c\u043e\u0433\u0443 \u043f\u043e\u043d\u044f\u0442\u044c - https://github.com/heartexlabs/label-studio/pull/157 \u0442\u0443\u0442 \u0431\u044b\u043b \u0435\u0449\u0435 \u043e\u0434\u0438\u043d \u043a\u043e\u043c\u043c\u0438\u0442 \u043a\u0443\u0434\u0430 \u043e\u043d \u043f\u0440\u043e\u0441\u0440\u0430\u043b\u0441\u044f\nnik \u00a02:48 PM MSK\u00a0uploaded this image: image.png\n\u0441\u043b\u0443\u0448\u0430\u0439\u0442\u0435 \u0430 \u0435\u0441\u043b\u0438 \u0443 \u043c\u0435\u043d\u044f \u0442\u0430\u043a\u0430\u044f \u043e\u0448\u0438\u0431\u043a\u0430 \u0432\u043e\u0437\u043d\u0438\u043a\u0430\u0435\u0442 LSB is not defined - \u044d\u0442\u043e \u043a\u0443\u0434\u0430 \u043a\u043e\u043f\u0430\u0442\u044c?"

    return text, spans


def data_for_test_2():
    text = u"\u043e\u0445 \u0431\u043b\u044f\n\u044f \u0447\u0435 \u0442\u043e \u0443\u0436\u0435 \u0437\u0430\u043f\u0443\u0442\u0430\u043b\u0441\u044f \u043a\u0443\u0434\u0430 \u0447\u0442\u043e \u043c\u0435\u0440\u0436\u0438\u0442\u044c\n\u043d\u0430\u0432\u0435\u0440\u043d\u043e\u0435 \u043c\u043d\u0435 \u044d\u0442\u043e \u043d\u0430\u0434\u043e \u0441\u0434\u0435\u043b\u0430\u0442\u044c \u0441\u0435\u0439\u0447\u0430\u0441\n\u0430 \u0442\u043e \u044f \u0442\u0443\u0442 \u0434\u043e\u0431\u0430\u0432\u043b\u044e \u043a\u043e\u0434\u0430 \u0434\u043b\u044f \u043a\u043e\u043d\u0444\u0438\u0433\u0430 \u0438 \u0432\u043e\u043e\u0431\u0449\u0435 \u0443\u043c\u0440\u0443"
    span = [
        {
            "end": 81,
            "label": "Task",
            "start": 68,
            "text": "\u0441\u0434\u0435\u043b\u0430\u0442\u044c \u0441\u0435\u0439\u0447\u0430\u0441"
        },
        {
            "end": 44,
            "label": "Task",
            "start": 29,
            "text": "\u043a\u0443\u0434\u0430 \u0447\u0442\u043e \u043c\u0435\u0440\u0436\u0438\u0442\u044c"
        },
        {
            "end": 5,
            "label": "EmotionDOWN",
            "start": 3,
            "text": "\u0431\u043b\u044f"
        },
        {
            "end": 110,
            "label": "Task",
            "start": 111
        },
        {
            "end": 131,
            "label": "EmotionDOWN",
            "start": 128,
            "text": "\u0443\u043c\u0440\u0443"
        }
    ]
    return text, span


def test_span_labeled_text_dataset():
    text, spans = data_for_test_2()
    s = SpanLabeledTextDataset(
        [text], [spans],
        tokenizer=BertTokenizer.from_pretrained(
            'bert-base-multilingual-cased',
            cache_dir=os.path.join(os.path.dirname(__file__), 'model', 'pretrained_model_cache')
        )
    )
    tokens, labels = s.list_of_tokens[0], s.list_of_labels[0]
    assert len(tokens) == len(labels)
    for token, label in zip(tokens, labels):
        print(f'{token} [{label}]')
    print(s)


if __name__ == "__main__":
    test_span_labeled_text_dataset()
