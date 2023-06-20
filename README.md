# Label Studio for Hugging Face's Transformers

[Website](https://labelstud.io/) • [Docs](https://labelstud.io/guide) • [Twitter](https://twitter.com/heartexlabs) • [Join Slack Community <img src="https://app.heartex.ai/docs/images/slack-mini.png" width="18px"/>](https://slack.labelstud.io/?source=github-1)

<br/>

**Transfer learning for NLP models by annotating your textual data without any additional coding.**

This package provides a ready-to-use container that links together:

- [Label Studio](https://github.com/heartexlabs/label-studio) as annotation frontend
- [Hugging Face's transformers](https://github.com/huggingface/transformers) as machine learning backend for NLP

<br/>

[<img src="https://raw.githubusercontent.com/heartexlabs/label-studio-transformers/master/images/codeless.png" height="500">](https://github.com/heartexlabs/label-studio-transformers)

### Quick Usage

#### Install Label Studio and other dependencies

```bash
pip install -r requirements.txt
```

##### Create ML backend with BERT classifier
```bash
label-studio-ml init my-ml-backend --script models/bert_classifier.py
cp models/utils.py my-ml-backend/utils.py

# Start ML backend at http://localhost:9090
label-studio-ml start my-ml-backend

# Start Label Studio in the new terminal with the same python environment
label-studio start
```

1. Create a project with `Choices` and `Text` tags in the labeling config.
2. Connect the ML backend in the Project settings with `http://localhost:9090`

##### Create ML backend with BERT named entity recognizer
```bash
label-studio-ml init my-ml-backend --script models/ner.py
cp models/utils.py my-ml-backend/utils.py

# Start ML backend at http://localhost:9090
label-studio-ml start my-ml-backend

# Start Label Studio in the new terminal with the same python environment
label-studio start
```

1. Create a project with `Labels` and `Text` tags in the labeling config.
2. Connect the ML backend in the Project settings with `http://localhost:9090`

#### Training and inference

The browser opens at `http://localhost:8080`. Upload your data on **Import** page then annotate by selecting **Labeling** page.
Once you've annotate sufficient amount of data, go to **Model** page and press **Start Training** button. Once training is finished, model automatically starts serving for inference from Label Studio, and you'll find all model checkpoints inside `my-ml-backend/<ml-backend-id>/` directory.

[Click here](https://labelstud.io/guide/ml.html) to read more about how to use Machine Learning backend and build Human-in-the-Loop pipelines with Label Studio

## License

This software is licensed under the [Apache 2.0 LICENSE](/LICENSE) © [Heartex](https://www.heartex.com/). 2020

<img src="https://github.com/heartexlabs/label-studio/blob/master/images/opossum_looking.png?raw=true" title="Hey everyone!" height="140" width="140" />
