## Introduction 

Implementation of the paper:
**Masked Language Model Scoring**

url:  https://arxiv.org/pdf/1910.14659.pdf

## Dependencies
PyTorch version >= 1.0.0
Python version >= 3.6

## Install fairseq by source
cd fairseq
pip install --editable .

## Basic settings
```python
import torch
# list available models
torch.hub.list("pytorch/fairseq") # will list ['bart.base', 'bart.large', 'bart.large.cnn', 'bart.large.mnli', 'bart.large.xsum', 'bpe', 'camembert', 'camembert-base', 'camembert-base-ccnet', 'camembert-base-ccnet-4gb', 'camembert-base-oscar-4gb', 'camembert-base-wikipedia-4gb', 'camembert-large', 'camembert.v0', 'conv.stories', 'conv.stories.pretrained', 'conv.wmt14.en-de', 'conv.wmt14.en-fr', 'conv.wmt17.en-de', 'data.stories', 'dynamicconv.glu.wmt14.en-fr', 'dynamicconv.glu.wmt16.en-de', 'dynamicconv.glu.wmt17.en-de', 'dynamicconv.glu.wmt17.zh-en', 'dynamicconv.no_glu.iwslt14.de-en', 'dynamicconv.no_glu.wmt16.en-de', 'lightconv.glu.wmt14.en-fr', 'lightconv.glu.wmt16.en-de', 'lightconv.glu.wmt17.en-de', 'lightconv.glu.wmt17.zh-en', 'lightconv.no_glu.iwslt14.de-en', 'lightconv.no_glu.wmt16.en-de', 'roberta.base', 'roberta.large', 'roberta.large.mnli', 'roberta.large.wsc', 'tokenizer', 'transformer.wmt14.en-fr', 'transformer.wmt16.en-de', 'transformer.wmt18.en-de', 'transformer.wmt19.de-en', 'transformer.wmt19.de-en.single_model', 'transformer.wmt19.en-de', 'transformer.wmt19.en-de.single_model', 'transformer.wmt19.en-ru', 'transformer.wmt19.en-ru.single_model', 'transformer.wmt19.ru-en', 'transformer.wmt19.ru-en.single_model', 'transformer_lm.gbw.adaptive_huge', 'transformer_lm.wiki103.adaptive', 'transformer_lm.wmt19.de', 'transformer_lm.wmt19.en', 'transformer_lm.wmt19.ru', 'xlmr.base', 'xlmr.large']
# we will import masekd Language Models (For example, roberta)

```

# Scoring
To run scoring, please follow the below command:

```bash
python score.py --PATH examples/asr-librispeech-espnet/data/dev-clean.am.json  --SAVE_PATH 3 --GROUP 2 --SUBSET 1
```


