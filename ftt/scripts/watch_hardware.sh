#!/bin/bash

# Script monitors usage of RAM, GPU RAM and GPU Power every 0.1 seconds.
watch -n .1 '
cpu_ram=$(ps -u $(whoami) --no-headers -o rss | awk "{sum+=\$1} END {print sum/1024 \" MiB\"}")
gpu_ram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "{sum+=\$1} END {print sum \" MiB\"}")
power_draw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | awk "{sum+=\$1} END {print sum \" W\"}")
process_info=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
printf "| RAM            | %s\n" "$cpu_ram"
printf "| GPU RAM        | %s\n" "$gpu_ram"
printf "| GPU Power Draw | %s\n" "$power_draw"
printf "| Process Info   | %s\n" "$process_info"
'