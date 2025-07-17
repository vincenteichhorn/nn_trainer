#!/bin/bash

# Script to tail watch log gile

BASE_DIR=/sc/projects/sci-herbrich/chair/lora-bp/vincent.eichhorn/ba/_jobs

# Check if the job ID is provided as an argument
if [ $# -eq 0 ]; then
    echo "No job ID provided. Usage: $0 <job_id>"
    exit 1
fi
JOB_ID=$1
# find file containing job id
LOG_FILE=$(find $BASE_DIR -name "*$JOB_ID*.log" | head -n 1)
# Check if the log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found for job ID: $JOB_ID"
    exit 1
fi
# Tail the log file
echo "Tailing log file: $LOG_FILE"
tail -F "$LOG_FILE"