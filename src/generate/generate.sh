for DATASET in korean_learner # native korean_learner lang8
do
	for SUBSET_TYPE in train test valid
	do
			for HYPERPARAM in dr1lr30 #dr1lr10  dr1lr20  dr1lr30  dr1lr40  dr1lr50  dr2lr10  dr2lr20  dr2lr30  dr2lr40  dr2lr50
			do
					FILE=./../../data/generated/$DATASET/${HYPERPARAM}_${SUBSET_TYPE}.txt
					if [ ! -f "$FILE" ]; then
							MODEL=./../../checkpoint/bart/$DATASET/$HYPERPARAM
							if [ -d "$MODEL" ]; then
									CUDA_VISIBLE_DEVICES=3 fairseq-generate \
									../../data/binarized/$DATASET \
									--user-dir ./../fairseq/models \
									--gen-subset $SUBSET_TYPE \
									--path ../../checkpoint/bart/$DATASET/$HYPERPARAM/checkpoint_best.pt \
									--batch-size 2048 --beam 5 --nbest 1 --remove-bpe \
									--max-tokens 6000 --no-early-stop >> ./../../data/generated/$DATASET/${HYPERPARAM}_${SUBSET_TYPE}.txt
					
							fi
					fi	
			done
	done
done
	
