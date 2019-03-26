#!/usr/bin/env bash

FULL_PTH=$1

echo "Copying $FULL_PTH
To s3://lexisnexis-news-incremental$FULL_PTH "

case "$FULL_PTH" in
        /*) pathchk -- "$FULL_PTH" && aws s3 cp "$FULL_PTH" s3://lexisnexis-news-incremental"$FULL_PTH";;
        *) echo "Bad path: $FULL_PTH";;
esac
