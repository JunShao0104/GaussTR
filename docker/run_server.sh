# Please accordingly change the following content before running:
# 1. TODO: /home/lzhao360: your home directory
# 2. /mnt: the directory you want to mount
# 3. /scratch: the directory you want to use for the dataset
# 4. TODO: lingjunz_occgaussian: the container name
# 5. occgaussian:latest: the image name

#!/usr/bin/env bash
USER_ID=$(id -u)
GROUP_ID=$(id -g)
PASSWD_FILE=$(mktemp) && echo $(getent passwd $USER_ID) > $PASSWD_FILE
GROUP_FILE=$(mktemp) && echo $(getent group $GROUP_ID) > $GROUP_FILE

docker run -it \
    -e HOME \
    -u $USER_ID:$GROUP_ID \
    -v $PASSWD_FILE:/etc/passwd:ro \
    -v $GROUP_FILE:/etc/group:ro \
    -v /nethome/lzhao360:$HOME \
    -v /mnt:/mnt \
    -v /scratch:/scratch \
    --name lingjunz_gausstr_cuda121 \
    --gpus=all \
    --ipc=host \
    gausstr_cuda121:latest # Your image name: can directly used the existing image