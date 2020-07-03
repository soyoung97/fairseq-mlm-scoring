fairseq-train \
    ./../../data/binarized/korean_learner \
    --user-dir ./../fairseq/models \
    --arch simple_lstm_seq2seq \
    --lr 1e-4 --clip-norm 0.1 --max-tokens 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler fixed \
    --save-dir ./../../checkpoint/LSTM/korean_learner