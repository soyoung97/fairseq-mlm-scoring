# Introduction 

Implementation of the paper:
**Masked Language Model Scoring**

url:  https://arxiv.org/pdf/1910.14659.pd.
Librispeech datasets are cloned from [librispeech\_link]

I implemented the following:
* Pre-processing of Librispeech dataset
* pppl scoring by re-inferencing the n-best generated output
* Rescoring
* WER(word error rate) calculating 


# Dependencies
 Python version >= 3.6
* [jiwer]==2.1.0 (For calculating Word Error Rate)
*  pytorch version==1.4.0

To automatically install dependencies, please refer to requirements.txt.


# Implementation

### pppl scoring (section 2.1 & 2.3 in the paper)
I used masked Language Models by fairseq. (For example, roberta).

Within dev-clean, dev-other, test-clean, and test-other, please run:
```bash  
mkdir models # you should create models directory first!
python score.py --ORIG_PATH examples/asr-librispeech-espnet/data/{CHANGE_THIS_PART}.am.json --PPPL_SCORE
``` 
It is slow - needs batching & optimization for faster speed.

### Sequence-to-sequence rescoring (section 3 in the paper)

The commands are similar to pppl scoring. You must run pppl scoring first and get the pppl-scored dataset before you run rescoring.
```bash
python score.py --ORIG_PATH examples/asr-librispeech-espnet/data/{CHANGE_THIS_PART}.am.json\
--PPPL_PATH examples/asr-librispeech-espnet/data/{CHANGE_THIS_PART}.am.pppl.json --RESCORE
```
This will also give you WER percentage.

### Overall
To run the whole process,
please run:
```
bash scripts/run.sh
```




[jiwer]: <https://pypi.org/project/jiwer/>
[librispeech\_link]: <https://github.com/awslabs/mlm-scoring/tree/master/examples/asr-librispeech-espnet>
