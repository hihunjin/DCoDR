dataset_name=celeba_9cls_39ctt

sudo -H -E PYTHONPATH=$PYTHONPATH:$pwd CUDA_VISIBLE_DEVICES=1 \
python \
evaluation.py \
--base-dir cache \
--eval-type prediction \
--eval-name clip \
--evaluated-exp-names '[]' \
--root-exps "['clip']" \
--train-data-name "$dataset_name"_train \
--delete-weights-folder False \
--chosen-epoch "last"
# --chosen-epoch "[20, 40, 60, 80, 100, 120, 140, 160, 180, 'last']"
