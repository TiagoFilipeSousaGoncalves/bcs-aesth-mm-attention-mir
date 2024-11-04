#!/bin/bash
#SBATCH --partition=gpu_min12gb
#SBATCH --qos=gpu_min12gb
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=DeiT_Base_Patch16_224.out
#SBATCH --error=DeiT_Base_Patch16_224.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
python src/main_image.py \
 --gpu_id 0 \
 --config_json 'config/image/DeiT_Base_Patch16_224.json' \
 --images_resized_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/breloai-rsz' \
 --images_original_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db' \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles' \
 --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results'
echo "Finished"
