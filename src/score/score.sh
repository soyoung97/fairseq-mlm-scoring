for DATASET in lang8 # native korean_learner lang8
do
	rm ./../../data/generated/$DATASET/score.txt 
	for SUBSET_TYPE in train test valid
	do
			for HYPERPARAM in dr2lr20 dr1lr20 #dr1lr10  dr1lr20  dr1lr30  dr1lr40  dr1lr50  dr2lr10  dr2lr20  dr2lr30  dr2lr40  dr2lr50
			do
					FILE=./../../data/generated/$DATASET/${HYPERPARAM}_${SUBSET_TYPE}.txt
					printf "\n\n${HYPERPARAM}_${SUBSET_TYPE}:\n" # >> ./../../data/generated/$DATASET/score.txt 
					if [ -f "$FILE" ]; then
							python ./../data/score.py --path ${FILE} # >> ./../../data/generated/$DATASET/score.txt
					fi	
			done
	done
done
	
