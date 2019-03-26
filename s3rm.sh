#!/usr/bin/env bash

FULL_PTH=$1

echo "Removing s3://lexisnexis-news-incremental$FULL_PTH "

aws s3 rm s3://lexisnexis-news-incremental"$FULL_PTH";
