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
 --images_path '' \
 --csvs_path '' \
 --results_dir 'results' \
 --verbose
echo "TBD"
