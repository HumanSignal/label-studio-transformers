# Label Studio for Transformers

[Website](https://labelstud.io/) • [Docs](https://labelstud.io/guide) • [Twitter](https://twitter.com/heartexlabs) • [Join Slack Community <img src="https://go.heartex.net/docs/images/slack-mini.png" width="18px"/>](https://docs.google.com/forms/d/e/1FAIpQLSdLHZx5EeT1J350JPwnY2xLanfmvplJi6VZk65C2R4XSsRBHg/viewform?usp=sf_link)

<br/>

**Transfer learning for NLP models by annotating your textual data without any additional coding.**

This package provides a ready-to-use container that links together:

- [Label Studio](https://github.com/heartexlabs/label-studio) as annotation frontend
- [Hugging Face's transformers](https://github.com/huggingface/transformers) as machine learning backend for NLP

<br/>

### Quick Usage

1. Start docker container:
    ```bash
    docker-compose up
    ```

2. Run `http://localhost:8200/` in browser, upload your data on [**Import** page](http://localhost:8200/import) and then do [**Labeling**](http://localhost:8200/).

3. Once you've finished, you can find all trained checkpoints in `storage/model/model` directory,
or use Label Studio API to send prediction request to deployed model:
    ```bash
    curl -X POST -H 'Content-Type: application/json' -d '{"text": "Its National Donut Day."}' http://localhost:8200/predict
    ```

### Advanced Usage

#### Select specific model
You can set a pre-trained transformer model from the [list](https://huggingface.co/models):
```bash
export pretrained_model=bert-base-uncased
```

#### View training logs
View training logs in console:
```bash
docker exec -it label-studio-ml-backend sh -c "tail -n100 /tmp/rq.log"
```

#### Run tensorboard:
```bash
tensorboard --logdir=storage/model/train_logs
```

## License

This software is licensed under the [Apache 2.0 LICENSE](/LICENSE) © [Heartex](https://www.heartex.ai/). 2020

<img src="https://github.com/heartexlabs/label-studio/blob/master/images/opossum_looking.png?raw=true" title="Hey everyone!" height="140" width="140" />
