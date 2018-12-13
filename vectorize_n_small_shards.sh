#!/usr/bin/env bash
printf "\n
Arg1: n shards to process
Arg2: input file dir
Arg3: output index dir \n"

printf "\nPreprocessing $1 shard(s) with small Universal Sentence Encoder\n";
for i in `seq 1 $1`
do
	printf "\n\nStarting $i/$1 @ $(date)\n";
	python -u py_scripts/preprocessing/prep_shard.py -i $2 -o $3 \
	-b base_indexes/USE_lite_base_IVF16K.index \
	-p progress.txt -t 2 -r -d;
done

printf "\n\n\nCompleted @ $(date)\n";
