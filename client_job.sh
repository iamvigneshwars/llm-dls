#!/bin/bash
#SBATCH --partition=cs05r
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=1
#SBATCH --mem=20GB
#SBATCH --nodelist=cs05r-sc-hop01-04
#SBATCH --job-name=llm-api
#SBATCH --output=llm-api_%j.log
#SBATCH --error=llm-api_%j.err
#SBATCH --time=30-00:00:00

export OLLAMA_MODELS=/dls/tmp/mrg27357/dev
ollama serve > ollama.log 2>&1 & 
sleep 5
source /dls/science/users/mrg27357/llm-dls/llm-dls/bin/activate
python /dls/science/users/mrg27357/llm-dls/llm_api.py --pdf /dls/science/users/mrg27357/llm-dls/data/user.pdf
echo "Job finished at: $(date)"
