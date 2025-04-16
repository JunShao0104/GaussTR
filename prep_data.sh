#!/bin/bash

# for pace
# DATA="/storage/coda1/p-lgan31/0/shared/dataset"

# for lnar-server
# DATA="/scratch/LNAR/dataset"
# DATA_OCC3D="/scratch/LNAR/dataset/Occ3d"

# for lnar-server-l40s
DATA="/scratch/shared/dataset"
DATA_OCC3D="/scratch/shared/dataset/Occ3d"

# for skynet
# DATA="/coc/flash5/datasets"
# DATA1="/coc/flash5/lzhao360"

# for workstation
# DATA="/home/lzhao360/Project/dataset"

# Create 'data' directory if it doesn't exist
if [ ! -d ./data ]; then
    mkdir ./data
    echo "Created './data' directory"
fi

# Create symbolic links for datasets
# Check if the target directory exists before creating the symbolic link
# Linking nuScenes dataset
if [ -d "$DATA/nuscenes/v1.0-full" ]; then
    ln -s "$DATA/nuscenes/v1.0-full" ./data/nuscenes
    echo "Linked nuScenes dataset to ./data/nuscenes"
else
    echo "Error: $DATA/nuscenes does not exist"
fi

if [ -d "$DATA_OCC3D/gts" ]; then
    ln -s "$DATA_OCC3D/gts" ./data/nuscenes/gts
    echo "Linked occ3d gt to ./data/nuscenes/gts"
else
    echo "Error: $DATA_OCC3D/gts does not exist"
fi

# ln -s /scratch/LNAR/dataset/nuscenes/v1.0-full ./data/nuscenes
# ln -s /scratch/LNAR/dataset/SurroundOcc/gaussianformer ./data/nuscenes_cam
# ln -s /scratch/LNAR/dataset/SurroundOcc/nuscenes_occ ./data/surroundocc