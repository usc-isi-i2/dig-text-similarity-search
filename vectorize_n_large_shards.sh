#!/usr/bin/env bash
printf "\n
Arg1: n shards to process: $1
Arg2: input file dir:      $2
Arg3: output index dir:    $3
Arg4: using progress file: ${4:progress.txt}
\n"

printf "\nPreprocessing $1 shard(s) with large Universal Sentence Encoder\n";
for i in `seq 1 $1`
do
	printf "\n\nStarting $i/$1 @ $(date)\n";
	python -u py_scripts/preprocessing/prep_shard.py -i $2 -o $3 \
	-b base_indexes/USE_large_base_IVF4K_15M.index \
	-p $4 -t 2 -l -n 64 -r -d;
done

printf "\n\n\nCompleted @ $(date)\n";
