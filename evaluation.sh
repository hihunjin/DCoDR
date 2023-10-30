dataset_name=celeba_39cls_9ctt

sudo -H -E PYTHONPATH=$PYTHONPATH:$pwd CUDA_VISIBLE_DEVICES=0 \
python \
evaluation.py \
--base-dir cache \
--eval-type prediction \
--eval-name DCoDR__"$dataset_name" \
--evaluated-exp-names '[]' \
--root-exps "['DCoDR_"$dataset_name"']" \
--train-data-name celeba_39cls_15ctt_train \
--delete-weights-folder False \
--chosen-epoch "[20, 40, 60, 80, 100]"
# --chosen-epoch "[20, 40, 60, 80, 100, 120, 140, 160, 180, 'last']"
