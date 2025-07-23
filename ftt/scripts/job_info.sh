#!/bin/bash

me=$(whoami)
tmpfile=$(mktemp)

# Collect job data
squeue -h -u "$me" -o "%i %u %T %P %V %j %M" | while read -r jobid user state part sub_time name time; do
    est_start=$(scontrol show job "$jobid" | awk -F= '/StartTime=/{print $2}' | cut -d' ' -f1)
    time_limit=$(scontrol show job "$jobid" | awk -F= '/TimeLimit=/{print $3}' | cut -d' ' -f1)
    [[ $est_start == "Unknown" ]] && est_start="9999-12-31T23:59:59"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$jobid" "$user" "$state" "$part" "$sub_time" "$est_start" "$time" "$time_limit" "$name"
done | sort -k6 > "$tmpfile"

# Print header
printf "%-10s %-16s %-10s %-15s %-20s %-20s %-10s %-12s %-30s\n" \
  "JOBID" "USER" "STATE" "PARTITION" "SUBMIT_TIME" "START_TIME" "TIME" "TIME_LIMIT" "NAME"

# Print sorted table
while IFS=$'\t' read -r jobid user state part sub_time est_start time time_limit name; do
    [[ $est_start == "9999-12-31T23:59:59" ]] && est_start="Unknown"
    printf "%-10s %-16s %-10s %-15s %-20s %-20s %-10s %-12s %-30s\n" \
      "$jobid" "$user" "$state" "$part" "$sub_time" "$est_start" "$time" "$time_limit" "$name"
done < "$tmpfile"

rm -f "$tmpfile"
