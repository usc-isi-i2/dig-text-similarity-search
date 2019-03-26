#!/usr/bin/env bash

FULL_PTH=${1:-/faiss/faiss_index_shards/deployment_full/}

printf "\nListing contents of s3://lexisnexis-news-incremental$FULL_PTH \n
(Note: Default index dir = /faiss/faiss_index_shards/deployment_full/) \n\n"

aws s3 ls s3://lexisnexis-news-incremental"$FULL_PTH";
