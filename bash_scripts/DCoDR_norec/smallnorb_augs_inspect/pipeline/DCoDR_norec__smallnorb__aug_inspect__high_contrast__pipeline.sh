#!/bin/bash




# train
bash bash_scripts/DCoDR_norec/smallnorb_augs_inspect/train/DCoDR_norec__smallnorb__aug_inspect__high_contrast__train.sh

# eval
bash bash_scripts/DCoDR_norec/smallnorb_augs_inspect/eval/DCoDR_norec__smallnorb__aug_inspect__high_contrast__eval__pred.sh

