#!/bin/bash

me=$(whoami)
tmpfile=$(mktemp)

# Collect job data
squeue -h -u "$me" -o "%i %u %T %P %V %j %M" | while read -r jobid user state part sub_time name time; do
    job_info=$(scontrol show job "$jobid")
    est_start=$(awk -F= '/StartTime=/{print $2}' <<< "$job_info" | cut -d' ' -f1)
    time_limit=$(awk -F= '/TimeLimit=/{print $3}' <<< "$job_info" | cut -d' ' -f1)
    priority=$(awk -F= '/Priority=/{print $2}' <<< "$job_info" | cut -d' ' -f1)
    req_nodelist=$(awk -F= '/ReqNodeList=/{print $2}' <<< "$job_info" | cut -d' ' -f1)
    reason=$(awk -F= '/Reason=/{print $3}' <<< "$job_info" | cut -d' ' -f1)

    [[ $est_start == "Unknown" ]] && est_start="9999-12-31T23:59:59"
    [[ -z $req_nodelist ]] && req_nodelist="(any)"
    [[ -z $reason ]] && reason="(none)"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$jobid" "$state" "$part" "$sub_time" "$est_start" "$time" "$time_limit" "$priority" "$req_nodelist" "$reason" "$name"
done | sort -k5 > "$tmpfile"

# Print header
printf "%-6s %-8s %-15s %-20s %-20s %-10s %-12s %-10s %-25s %-20s %-20s\n" \
  "JOBID" "STATE" "PARTITION" "SUBMIT_TIME" "START_TIME" "TIME" "TIME_LIMIT" "PRIORITY" "REQ_NODELIST" "REASON" "NAME" 

# Print sorted table
while IFS=$'\t' read -r jobid state part sub_time est_start time time_limit priority req_nodelist reason name; do
    [[ $est_start == "9999-12-31T23:59:59" ]] && est_start="Unknown"
    reason_trunc=${reason:0:20}
    name_trunc=${name:0:20}
    printf "%-6s %-8s %-15s %-20s %-20s %-10s %-12s %-10s %-25s %-20s %-20s\n" \
      "$jobid" "$state" "$part" "$sub_time" "$est_start" "$time" "$time_limit" "$priority" "$req_nodelist" "$reason_trunc" "$name_trunc"
done < "$tmpfile"

rm -f "$tmpfile"
