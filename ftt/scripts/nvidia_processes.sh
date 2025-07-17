# run as root
# check all nvidia processes, check for user
# fuser -v /dev/nvidia*
# kill a nvidia python processes
# for i in $(sudo lsof /dev/nvidia0 | grep python  | awk '{print $2}' | sort -u); do kill -9 $i; done