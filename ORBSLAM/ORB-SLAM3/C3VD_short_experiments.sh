#!/bin/bash
pathToSequences="/home/richard/Datasets/C3VD/" #Example, it is necesary to change it by the dataset path
initial_index=0
executions=5
arrayExperiments=('cecum_t1_a' 'cecum_t1_b' 'cecum_t2_a' 'cecum_t2_b' 'cecum_t2_c' 'cecum_t3_a' 'cecum_t4_a' 'cecum_t4_b' 'desc_t4_a' 'sigmoid_t1_a' 'sigmoid_t2_a' 'sigmoid_t3_a' 'sigmoid_t3_b' 'trans_t1_a' 'trans_t1_b' 'trans_t2_a' 'trans_t2_b' 'trans_t2_c' 'trans_t3_a' 'trans_t3_b' 'trans_t4_a' 'trans_t4_b')


##########################
# CudaSIFT experiments 
##########################
pathBaseResults="$OUT_HOST/ORB-SLAM3_C3VD_shorts_2/"
if [[ ! -e $pathBaseResults ]]; then
    mkdir -p $pathBaseResults
elif [[ ! -d $pathBaseResults ]]; then
    echo "$pathBaseResults already exists but is not a directory" 1>&2
    exit
fi

##########################
# CudaSIFT experiments 
##########################

for folder in ${arrayExperiments[@]}; do
  pathSeq="$pathToSequences$folder"
  pathResult="$pathBaseResults$folder"
  if [[ ! -e $pathResult ]]; then
    mkdir -p $pathResult
  elif [[ ! -d $pathResult ]]; then
    echo "$pathResult already exists but is not a directory" 1>&2
    exit
  fi
  
  for i in $(seq 1 $executions)
  do
    echo "Running CudaSIFT $folder --> $i/$executions"
    ./Examples/Monocular/mono_endo_c3vd ./Vocabulary/ORBvoc.txt ./Examples/Monocular/C3VD_Endoscope_Depth.yaml "$pathSeq"
    index=$(( $initial_index + $i ))
    pathResult="$pathBaseResults$folder/$index"
    if [[ ! -e $pathResult ]]; then
      mkdir -p $pathResult
    elif [[ ! -d $pathResult ]]; then
      echo "$pathResult already exists but is not a directory" 1>&2
      exit
    fi
    
    shopt -s nullglob # To avoid error if the folder is empty
    mv output/* "$pathResult"
    shopt -u nullglob
  done
done
