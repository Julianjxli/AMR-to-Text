# AMR-to-Text

run command:
CUDA_VISIBLE_DEVICES=3 python bin/train.py --config configs/config3.yaml --direction text


CUDA_VISIBLE_DEVICES=4  python bin/predict_sentences.py \
    --datasets    /data/home/lijx/Spring/data/data3.0/amrs/split/test/*.txt\
    --gold-path data/tmp/gold.textdis1114.txt \
    --pred-path data/tmp/pred.textdis1114.txt \
    --checkpoint  /data/home/lijx/Spring/runs/24/best-bleu_checkpoint_8_41.7867.pt\
    --beam-size 1 \
    --batch-size 500 \
    --device cuda \
    --penman-linearization --use-pointer-tokens
