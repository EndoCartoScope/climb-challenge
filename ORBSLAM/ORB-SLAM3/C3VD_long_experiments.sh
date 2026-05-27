#!/bin/bash
pathToSequences="/home/richard/Datasets/C3VD_long/" #Example, it is necesary to change it by the dataset path
initial_index=0
executions=5
arrayExperiments=('seq1' 'seq2' 'seq3' 'seq4')

pathBaseResults="/home/richard/Experiments/ORB-SLAM3_C3VD_long_deinter/"
if [[ ! -e $pathBaseResults ]]; then
    mkdir -p $pathBaseResults
elif [[ ! -d $pathBaseResults ]]; then
    echo "$pathBaseResults already exists but is not a directory" 1>&2
    exit
fi

##########################
# Experiments 
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
    echo "Running ORB-SLAM3 $folder --> $i/$executions"
    ./Examples/Monocular/mono_endo_c3vd ./Vocabulary/ORBvoc.txt ./Examples/Monocular/C3VD_Endoscope_Depth.yaml "$pathSeq"/images_deinter
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
