dataset_name=celeba_9cls_15ctt

sudo -H -E PYTHONPATH=$PYTHONPATH:$pwd CUDA_VISIBLE_DEVICES=0 \
python \
main.py \
--base-dir cache \
--cuda True \
--num-workers 16 \
--load-weights-exp debug \
--load-weights-epoch last \
--exp-name DCoDR_clip_"$dataset_name" \
--data-name "$dataset_name"_train \
--test-data-name "$dataset_name"_test \
--batch-size 256 \
--epochs 120 \
--content-dim 128 \
--class-dim 256 \
--use-pretrain False  \
--enc-arch moco_resnet50 \
DCoDR_multi_arg \
--use-fc-head False \
--tau 0.2 \
--num-pos 1 \
--num-rand-negs 64 \
--class-negs True \
--num-b-cls-samp 128 \
--num-b-cls 2 \
--reconstruction-decay 0.3 \
--shifting-key tau \
--shifting-args '[0.2]'