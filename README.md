# Label Studio bundle for Transformers

This package provides a ready-to-use container that links together:

- [Label Studio](https://github.com/heartexlabs/label-studio) as annotation frontend,
- [Hugging Face's transformers](https://github.com/huggingface/transformers) as machine learning backend for NLP.

This allows you to learn very powerful NLP models simply by annotating your textual data without any additional coding.


### Usage
1. **input data**: define input text items that should be annotated, e.g. line-by-line text (various formats [also supported](https://labelstud.io/guide/format.html#Input)). 
Put your file(s) with input_data into `storage/label-studio/` directory (see [example](storage/label-studio/example.txt)).
Create environmental variable with `/data` prefix replacing `storage/label-studio`, i.e.:
    ```bash
    export input_data=/data/example.txt
    ```
    
2. **labels**: define labeling configuration scheme (_config_) with your labels you want to assign (see [example](storage/label-studio/label.xml)).
Put this file into the same directory `storage/label-studio` and also expose variable:
    ```bash
    export config=/data/labels.xml
    ```

3. **pretrained model**: use a pretrained transformer from the [list](https://huggingface.co/models). 
Create environmental variable with model:
    ```bash
    export pretrained_model=bert_base_uncased
    ```

4. Start docker container with environmental variables:
    ```bash
    docker-compose up
    ```

5. Go to `http://localhost:8200` and annotate your data in Label Studio interface. Once you finished, you can find all trained checkpoints in `model/` directory,
or use Label Studio API to send prediction request:
    ```bash
    curl -X POST -H 'Content-Type: application/json' -d '{"text": "Its National Donut Day."}' http://localhost:8200/predict
    ```
 
Happy labelling!