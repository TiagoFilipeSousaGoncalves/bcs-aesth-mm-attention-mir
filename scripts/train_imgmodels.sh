#!/bin/bash
#SBATCH --partition=debug_8gb
#SBATCH --qos=debug_8gb
#SBATCH --job-name=imgmodels
#SBATCH --output=imgmodels.out
#SBATCH --error=imgmodels.err



echo "TBD"
python src/main_image.py \
 --gpu_id 0 \
 --config_json 'config/config_image.json' \
 --images_resized_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/breloai-rsz' \
 --images_original_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db' \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles' \
 --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results' \
 --verbose
echo "TBD"
