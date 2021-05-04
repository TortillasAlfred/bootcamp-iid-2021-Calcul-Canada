#!/bin/bash
#SBATCH --account=def-lulam50                                 # Account with resources
#SBATCH --cpus-per-task=1                                     # Number of CPUs
#SBATCH --gres=gpu:1                                          # Number of GPUs (per node)
#SBATCH --mem=5G                                              # memory (per node)
#SBATCH --time=0-00:30                                        # time (DD-HH:MM)
#SBATCH --mail-user=mathieu.godbout.3@ulaval.ca               # Where to email
#SBATCH --mail-type=FAIL                                      # Email when a job fails
#SBATCH --output=/scratch/magod/%A_%a.out                     # Default write output on scratch, to jobID_arrayID.out file
#SBATCH --array=1-20                                          # Launch an array of 20 jobs

# Activate the Python environment
source /home/magod/venvs/CC_example/bin/activate

# Launch the main script
# You can access the array ID via $SLURM_ARRAY_TASK_ID
python main_script.py --dataset_path /project/def-adurand/magod/boston_dataset/ --logging_path /scratch/magod/CC_example --job_index $SLURM_ARRAY_TASK_ID
