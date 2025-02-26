#!/bin/bash
#SBATCH -J yejin
#SBATCH --output=eval.out

echo "### eval start"
echo "### START DATE=$(date)"

# modify the eval_config.py file
python eval_soft_voting.py

EOF