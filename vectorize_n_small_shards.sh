#!/usr/bin/env bash

n_shards=$1
input_dir=$2
output_dir=$3
progress_txt=${4:-progress.txt}


printf "\n
Arg1: n shards to process: $n_shards
Arg2: input file dir:      $input_dir
Arg3: output index dir:    $output_dir
Arg4: using progress file: $progress_txt
\n"


printf "\nPreprocessing $n_shards shard(s) with small Universal Sentence Encoder\n";

for i in `seq 1 "$n_shards"`
do
	printf "\n\nStarting $i/$n_shards @ $(date)\n";
	python -u py_scripts/preprocessing/prep_shard.py -i "$input_dir" -o "$output_dir" \
	-b base_indexes/USE_lite_base_IVF16K.index \
	-p "$progress_txt" -t 2 -r -d;
done

printf "\n\n\nCompleted @ $(date)\n\n";
