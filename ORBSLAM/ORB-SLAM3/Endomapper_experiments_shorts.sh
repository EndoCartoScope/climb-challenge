#!/bin/bash
initial_index=0
executions=5

pathResults="/home/richard/Experiments/ORB-SLAM3_endomapper_short/"
pathToSequences="/home/richard/Datasets/all_sequences/"

# Create the results folder if it is necessary of give an error
if [[ ! -e $pathResults ]]; then
    mkdir -p $pathResults
elif [[ ! -d $pathResults ]]; then
    echo "$pathResults already exists but is not a directory" 1>&2
    exit
fi

Sequences=(
  "Seq_001_a.mp4" "Seq_001_c.mp4" "Seq_002_b.mp4" "Seq_003_b.mp4" "Seq_006_111.mp4"
  "Seq_009_143.mp4" "Seq_011_65.mp4" "Seq_011_74.mp4" "Seq_016_d.mp4" "Seq_016_e.mp4"
  "Seq_020_b.mp4" "Seq_027_d.mp4" "Seq_035_a.mp4" "Seq_035_c.mp4" "Seq_035_d.mp4"
  "Seq_035_e.mp4" "Seq_050_73.mp4" "Seq_050_83.mp4" "Seq_050_87.mp4" "Seq_050_90.mp4"
)

##########################
# Experiments
##########################
for video_seq in ${Sequences[@]}; do
    pathVideo="$pathToSequences$video_seq"
    name_seq="${video_seq%.*}"
    pathResultSeq="$pathResults$name_seq"
    if [[ ! -e $pathResultSeq ]]; then
        mkdir -p $pathResultSeq
    elif [[ ! -d $pathResultSeq ]]; then
        echo "$pathResultSeq already exists but is not a directory" 1>&2
        exit
    fi
  
    for i in $(seq 1 $executions); do
        echo "Running $name_seq --> $i/$executions"
        echo "Video path: $pathVideo"
    	./Examples/Monocular/mono_endo_hculb ./Vocabulary/ORBvoc.txt ./Examples/Monocular/HCULB_Endoscope_Depth.yaml "$pathVideo"
        
        index=$(( $initial_index + $i ))
        pathResultExp="$pathResultSeq/$index"
        if [[ ! -e $pathResultExp ]]; then
            mkdir -p $pathResultExp
        elif [[ ! -d $pathResultExp ]]; then
            echo "$pathResultExp already exists but is not a directory" 1>&2
            exit
        fi
        
        shopt -s nullglob # To avoid error if the folder is empty
        mv output/* "$pathResultExp"
        shopt -u nullglob
    done
done

