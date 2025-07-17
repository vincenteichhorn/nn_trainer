
#!/bin/bash
# Description: Enroot configuration file for the LoRa BP container
# Environment Variables:
# - SSH_PORT: SSH port to use (default: 10022)
# - WORKSPACE_DIR: Workspace directory (default: /workspace)
# - USE_GROUP_SHARE: Enable/disable group share mounting (default: true)
#
# Command-line flags:
# -g, --no-group-share: Disable group share mounting

GROUP_SHARE="/sc/projects/sci-herbrich/chair/lora-bp"

mounts() {
    echo "${HOME} ${HOME}"
    # Only mount the group share if USE_GROUP_SHARE is true
    if [ "${USE_GROUP_SHARE}" = "true" ]; then
        echo "${GROUP_SHARE} ${GROUP_SHARE}"
    fi
}

hooks() {
    echo "SSH_PORT=${SSH_PORT:-10022}" >> ${ENROOT_ENVIRON}
    echo "WORKSPACE_DIR=${WORKSPACE_DIR:-/workspace}" >> ${ENROOT_ENVIRON}
    echo "WORKSPACE_DIR_GROUP=${WORKSPACE_DIR:-/workspace-group}" >> ${ENROOT_ENVIRON}
    echo "GROUP_SHARE=${GROUP_SHARE}" >> ${ENROOT_ENVIRON}
    echo "USE_GROUP_SHARE=${USE_GROUP_SHARE:-true}" >> ${ENROOT_ENVIRON}
    echo "EXIT_CODE_FILE_CONTAINER=${EXIT_CODE_FILE_CONTAINER:-/mnt/home/vincent.eichhorn/jobs/ext.log}" >> ${ENROOT_ENVIRON}	
    echo "SCRIPT_IN_CONTAINER=${SCRIPT_IN_CONTAINER:-bash}" >> ${ENROOT_ENVIRON}
    echo "SCRIPT_ARGS=${SCRIPT_ARGS:-bash}" >> ${ENROOT_ENVIRON}
}

rc(){
    # Optional: Print the selected SSH port and workspace directory for debugging
    echo -e "\n"
    echo -e "\033[1;34m[====================================]\033[0m"
    echo -e "\033[1;32m  > YOU HAVE ENTERED THE CONTAINER <  \033[0m"
    echo -e "\n"
    echo -e "\033[1;33m  • Using SSH Port: \033[1;31m${SSH_PORT}\033[0m"
    echo -e "\033[1;33m  • Workspace Directory: \033[1;31m${WORKSPACE_DIR}\033[0m"
    echo -e "\033[1;33m  • Group Share: \033[1;31m${USE_GROUP_SHARE}\033[0m"
    echo -e "\033[1;34m[====================================]\033[0m"
    
    # Check if the workspace directory exists, if not create it and set up the symlink
    if [ ! -e "${WORKSPACE_DIR}" ]; then
        ln -s "${HOME}" "${WORKSPACE_DIR}"
    fi

    # Only set up the group workspace if USE_GROUP_SHARE is true
    if [ "${USE_GROUP_SHARE}" = "true" ] && [ ! -e "${WORKSPACE_DIR_GROUP}" ]; then
        ln -s "${GROUP_SHARE}" "${WORKSPACE_DIR_GROUP}"
    fi

    # Set HOME to the workspace directory and define working directory
    export HOME="${WORKSPACE_DIR}"
    WD=${HOME}/lora-bp

    # Start the SSH server
    /usr/sbin/sshd -p ${SSH_PORT}

    # Navigate to the working directory and start bash
   echo "Executing Script: "
   echo "$SCRIPT_IN_CONTAINER $SCRIPT_ARGS"
   cd "${WD}"
   $SCRIPT_IN_CONTAINER $SCRIPT_ARGS
   RC=$?

   echo "Script exited with code $RC"
   echo "Write to ext code file $EXIT_CODE_FILE_CONTAINER"
   echo $RC > ${EXIT_CODE_FILE_CONTAINER}

}
