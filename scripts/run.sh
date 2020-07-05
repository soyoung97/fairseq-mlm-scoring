mkdir models
for DATASET in dev-clean  dev-other  test-clean  test-other
do
		python score.py --ORIG_PATH examples/asr-librispeech-espnet/data/{$DATASET}.am.json --PPPL_SCORE
		python score.py --ORIG_PATH examples/asr-librispeech-espnet/data/{$DATASET}.am.json \
				--PPPL_PATH examples/asr-librispeech-espnet/data/{$DATASET}.am.pppl.json --RESCORE
