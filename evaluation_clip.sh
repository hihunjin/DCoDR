dataset_name=celeba_39cls_9ctt

sudo -H -E PYTHONPATH=$PYTHONPATH:$pwd CUDA_VISIBLE_DEVICES=3 \
python \
evaluation.py \
--base-dir cache \
--eval-type prediction \
--eval-name DCoDR__"$dataset_name"_clip \
--evaluated-exp-names '[]' \
--root-exps "['DCoDR_clip_"$dataset_name"']" \
--train-data-name celeba_39cls_15ctt_train \
--delete-weights-folder False \
--chosen-epoch "[20, 40]"
# --chosen-epoch "[20, 40, 60, 80, 100, 120, 140, 160, 180, 'last']"
# --chosen-epoch "[20, 40, 60]"
