#!/bin/bash

# $1 = Job name
# $2 = Host path to shell script (e.g., /sc/home/user/scripts/test.sh)
# $3 = Time limit (e.g., 01:00:00)
# $4 = Node list (e.g., gx28)
# $5 = Script arguments (optional)

JOB_NAME=$1
SCRIPT_ON_HOST=$2
TIME_LIMIT=$3
NODELIST=$4
SCRIPT_ARGS=$5
MAX_RERUNS=250
CURR_RERUN=0

# Validate script path
if [[ ! -f "$SCRIPT_ON_HOST" ]]; then
  echo "❌ Error: Provided script does not exist on host: $SCRIPT_ON_HOST"
  exit 1
fi

JOB_UUID=$(uuidgen)
echo "JOB UUID: $JOB_UUID"
echo "JOB NAME: $JOB_NAME"
echo "TIME LIMIT: $TIME_LIMIT"
echo "NODELIST: $NODELIST"
echo "SCRIPT ON HOST: $SCRIPT_ON_HOST"
echo "SCRIPT ARGS: $SCRIPT_ARGS"
EXIT_CODE_FILE_HOST="/sc/home/vincent.eichhorn/jobs/ext-$JOB_UUID.log"
EXIT_CODE_FILE_CONTAINER="/mnt${EXIT_CODE_FILE_HOST#/sc}"
export EXIT_CODE_FILE_CONTAINER

# Translate host script path to container path: /sc → /mnt
SCRIPT_IN_CONTAINER="/mnt${SCRIPT_ON_HOST#/sc}"
export SCRIPT_IN_CONTAINER
export SCRIPT_ARGS

ENROOT_CONFIG="/sc/home/vincent.eichhorn/nn_trainer/ftt/scripts/enroot_config.sh"
IMAGE="hpi-artificial-intelligence-teaching/lora-bp-base-cuda-v2"
TAG="cuda-12.2"
FULL_IMAGE="ghcr.io/$IMAGE"
SQSH_FILE="$(basename $IMAGE)+$TAG.sqsh"
CONTAINER_NAME="$(basename $IMAGE)+$TAG.container"
USE_GROUP_SHARE=true
PARTITION="aisc"
export USE_GROUP_SHARE

sbatch <<EOT
#!/bin/bash -eux
#SBATCH --job-name=$JOB_NAME
#SBATCH --account sci-herbrich
#SBATCH --nodelist=$NODELIST
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=16
#SBATCH --mem=100000
#SBATCH --time=$TIME_LIMIT
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vincent.eichhorn@student.hpi.uni-potsdam.de
#SBATCH --output=/sc/projects/sci-herbrich/chair/lora-bp/vincent.eichhorn/nnt/_jobs/job_${JOB_NAME}-%j.log

CURR_RERUN=0

# Run the script inside the container, retry until it exits with 0
while true; do
  echo "=== Running script in container at \$(date) ==="
  enroot start --rw \\
    -c $ENROOT_CONFIG \\
    -m "/sc:/mnt" \\
    $CONTAINER_NAME \\
    $SCRIPT_IN_CONTAINER $SCRIPT_ARGS
  EXIT_CODE=\$(cat ${EXIT_CODE_FILE_HOST} || echo 0)
  echo "=== Script exited with code \$EXIT_CODE ==="
  
  if [ \$EXIT_CODE -eq 0 ]; then
    echo "Script finished successfully!"
    break
  else
    echo "Script failed with code \$EXIT_CODE. Retrying in 1 second..."
    CURR_RERUN=\$((CURR_RERUN + 1))
    if [ \$CURR_RERUN -ge $MAX_RERUNS ]; then
      echo "Maximum reruns reached (\$MAX_RERUNS). Exiting with failure."
      exit 1
    fi
    echo "Rerun \$CURR_RERUN of $MAX_RERUNS"
    echo "Waiting 1 second before rerun..."
    sleep 1
  fi
done
EOT
