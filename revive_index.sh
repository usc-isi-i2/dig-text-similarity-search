#!/usr/bin/env bash

# Specify target dirs to restore shards in
MAIN_IDXS="/faiss/faiss_index_shards/deployment_full/"
TMP_IDXS="/green_room/idx_deploy_B/"
S3_BKUP="s3://lexisnexis-news-incremental$MAIN_IDXS"

mkdir -p "$MAIN_IDXS"
mkdir -p "$TMP_IDXS"

n_main=$(ls "$MAIN_IDXS"*.index | wc -l)
n_tmp=$(ls "$TMP_IDXS"*.index | wc -l)
n_s3=$(aws s3 ls "$S3_BKUP" | wc -l)

# Echo n indexes
echo "Main: $MAIN_IDXS has $n_main shards"
echo "Tmp:  $TMP_IDXS has $n_tmp shards"
echo "S3:   $S3_BKUP has $n_s3 shards"

# Prompt user to agree to sync
read -p "Found $n_main existing shards. Restoring the index may overwrite them. Continue? [y/N]" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$MAIN_IDXS"
    aws s3 sync s3://lexisnexis-news-incremental/"$MAIN_IDXS" .
else
    exit 1
fi

# Prepare TMP Indexes
ACTIVATE="/home/ubuntu/anaconda3/bin/activate"
cd "$TMP_IDXS" && . "$ACTIVATE" dig_text_similarity

python -u /faiss/dig-text-similarity-search/py_scripts/preprocessing/consolidate_shards.py \
"$MAIN_IDXS" "$TMP_IDXS" -c -t 2        # Copy, N-Threads 2
