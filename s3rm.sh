#!/usr/bin/env bash

FULL_PTH=$1

printf "\nRemoving s3://lexisnexis-news-incremental$FULL_PTH \n\n"

aws s3 rm s3://lexisnexis-news-incremental"$FULL_PTH";
