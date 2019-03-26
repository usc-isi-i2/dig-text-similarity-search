#!/usr/bin/env bash

## Prep
# Constant working dirs
NEWS_DIR="/faiss/sage_news_data/raw/LN_WLFilt_extractions_IN/"
DONE_DIR="/faiss/sage_news_data/raw/LN_WLFilt_extractions_OUT/"
PUB_DATES="/faiss/sage_news_data/pub_date_split/"
DAILY_DIR="/faiss/faiss_index_shards/tmp_daily_ingest/"

PY_SCRIPTS="/faiss/dig-text-similarity-search/py_scripts/preprocessing/"
SERVICE="/faiss/dig-text-similarity-search/py_scripts/service/"

MAIN_IDXS="/faiss/faiss_index_shards/deployment_full/"
TMP_IDXS="/green_room/idx_deploy_B/"

# Get file to process
FILE=$(ls "${NEWS_DIR}" | head -1)

if [[ -z "$FILE" ]]; then
    echo "Nothing to process..."
    exit 1
else
    echo "Processing $FILE"
fi

# Get YYYY-MM-DD
YYYYMMDD=$(echo "$FILE" | grep -Eo "[0-9]{4}\-[0-9]{2}\-[0-9]{2}")

YYYY=$(echo "$YYYYMMDD" | cut -d "-" -f1)
MM=$(echo "$YYYYMMDD" | cut -d "-" -f2)
DD=$(echo "$YYYYMMDD" | cut -d "-" -f3)

echo "$YYYY  $MM  $DD"

# New working dirs
DATE_SPLIT="${PUB_DATES}${YYYY}_extraction_${MM}-${DD}/"
DAILY_IDXS="${DAILY_DIR}${YYYY}_indexes_${MM}-${DD}/"

if [[ -d "$DATE_SPLIT" ]] || [[ -d "$DAILY_IDXS" ]]; then
    echo "$FILE has already been processed"
    exit 1
else
    mkdir "$DATE_SPLIT"
    mkdir "$DAILY_IDXS"
fi


## Split
echo "Splitting $FILE by publication date..."
python -u "${PY_SCRIPTS}sort_by_pub_date.py" \
"${NEWS_DIR}${FILE}" "$DATE_SPLIT" \
-i "$(date -d "-45 days" -I)" -f "${YYYYMMDD}";


## Vectorize
n_shards=$(ls "${DATE_SPLIT}" | wc -l)
/faiss/dig-text-similarity-search/vectorize_n_large_shards.sh \
"$n_shards" "$DATE_SPLIT" "$DAILY_IDXS" "${DATE_SPLIT}progress.txt";


## Save backup (old)
BACKUP_DIR="/faiss/faiss_index_shards/backups/WL_${MM}${DD}"
python -u "${PY_SCRIPTS}consolidate_shards.py" "$DAILY_IDXS" "$BACKUP_DIR" --cp;


## Merge into indexes
# Switch to tmp service
kill -15 $(ps -ef | grep "[s]imilarity_server" | awk \'{print $2}\'); sleep 1;
python -u "${SERVICE}similarity_server.py" "$TMP_IDXS" -l -c 6 &

# Zip-merge into main indexes
cd "$MAIN_IDXS"
BEFORE=$(*.i*); cd -
echo "n" | python -u "${PY_SCRIPTS}consolidate_shards.py" \
"$DAILY_IDXS" "$MAIN_IDXS" --zip -p "zip_to_${MM}${DD}" -t 2;

cd "$MAIN_IDXS"
AFTER=$(*.i*); cd -

# Switch back to main service
LOG_FILE="/faiss/dig-text-similarity-search/logs/service/deploy_${MM}${DD}.out"
kill -15 $(ps -ef | grep "[s]imilarity_server" | awk \'{print $2}\'); sleep 1;
python -u "${SERVICE}similarity_server.py" "$MAIN_IDXS" -l -c 6 &

# Zip-merge into tmp
echo "y" | python -u "${PY_SCRIPTS}consolidate_shards.py" \
"$DAILY_IDXS" "$TMP_IDXS" --zip -p "zip_to_${MM}${DD}" -t 2;


## Cleanup
rm "${DATE_SPLIT}*.jl" "${DATE_SPLIT}*/*.jl"
rmdir "${DATE_SPLIT}old_news" "${DATE_SPLIT}date_error"

mv "${NEWS_DIR}${FILE}" "${DONE_DIR}${FILE}"


## Save backup (new)
B4=" ${BEFORE[*]} "
for item in ${AFTER[@]}; do
    if [[ $B4 =~ " $item " ]]; then
        echo "WIP"
    else
        echo "WIP"
    fi
done
