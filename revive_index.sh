#!/usr/bin/env bash

# Specify target dirs to restore shards in
MAIN_IDXS="/faiss/faiss_index_shards/deployment_full/"
TMP_IDXS="/green_room/idx_deploy_B/"
S3_BKUP="s3://lexisnexis-news-incremental$MAIN_IDXS"

mkdir -p "$MAIN_IDXS"
mkdir -p "$TMP_IDXS"


# Echo n backed up and existing indexes
n_main=$(ls "$MAIN_IDXS"*.i* | wc -l)
n_tmp=$(ls "$TMP_IDXS"*.i* | wc -l)
aws s3 ls "$S3_BKUP" --summarize --human-readable
echo "Main: $MAIN_IDXS has $n_main index items"
echo "Tmp:  $TMP_IDXS has $n_tmp index items"

# Prompt user to agree to sync
read -p "Continue sync? [y/N]" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$MAIN_IDXS"
    aws s3 sync s3://lexisnexis-news-incremental/"$MAIN_IDXS" .
else
    exit 1
fi

# cd TMP and remove existing indexes
cd "$TMP_IDXS" && rm *.i*
ACTIVATE="/home/ubuntu/anaconda3/bin/activate"
. "$ACTIVATE" dig_text_similarity

# Prepare TMP Indexes
DT_SIM="/faiss/dig-text-similarity-search/"
python -u "${DT_SIM}py_scripts/preprocessing/consolidate_shards.py" \
"$MAIN_IDXS" "$TMP_IDXS" -c -t 2        # Copy, N-Threads 2
