#!/bin/bash
#SBATCH --partition=gpu_min32gb
#SBATCH --qos=gpu_min32gb
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=MViTv2_Tiny.out
#SBATCH --error=MViTv2_Tiny.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
echo "Catalogue Type: E"
python src/main_multimodal.py \
 --gpu_id 0 \
 --config_json 'config/multimodal/E/MViTv2_Tiny.json' \
 --images_resized_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/breloai-rsz/E' \
 --images_original_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db' \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
 --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-11-13_01-54-16/bin/model_final.pt' \
 --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E'
echo "Finished"

echo "Catalogue Type: F"
python src/main_multimodal.py \
 --gpu_id 0 \
 --config_json 'config/multimodal/F/MViTv2_Tiny.json' \
 --images_resized_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/breloai-rsz/F' \
 --images_original_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db' \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
 --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-11-13_07-04-51/bin/model_final.pt' \
 --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F'
echo "Finished"
