#!/bin/bash
#SBATCH --partition=gpu_min80gb
#SBATCH --qos=gpu_min80gb
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=MaxViT_Tiny_224.out
#SBATCH --error=MaxViT_Tiny_224.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
echo "Catalogue Type: E"
python src/main_image.py \
 --gpu_id 0 \
 --config_json 'config/image/E/MaxViT_Tiny_224.json' \
 --images_resized_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/breloai-rsz/E' \
 --images_original_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db' \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
 --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E' \
 --train_or_test 'train'
echo "Finished"

echo "Catalogue Type: F"
python src/main_image.py \
 --gpu_id 0 \
 --config_json 'config/image/F/MaxViT_Tiny_224.json' \
 --images_resized_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/breloai-rsz/F' \
 --images_original_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db' \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
 --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F' \
 --train_or_test 'train'
echo "Finished"