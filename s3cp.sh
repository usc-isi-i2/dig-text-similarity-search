#!/usr/bin/env bash

FULL_PTH=$1

printf "\nCopying $FULL_PTH
To s3://lexisnexis-news-incremental$FULL_PTH \n
(Note: Requires /full/path/to/file.ext) \n\n"

case "$FULL_PTH" in
        /*) pathchk -- "$FULL_PTH" && aws s3 cp "$FULL_PTH" s3://lexisnexis-news-incremental"$FULL_PTH";;
        *) echo "Bad path: $FULL_PTH";;
esac
