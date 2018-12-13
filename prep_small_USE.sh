#!/usr/bin/env bash
# VersionArg: model version number (highest number will be deployed)
VersionArg=${1:"001"}

# Make Dockerized Model
python dt_sim/vectorizer/make_service_model.py -v "$VersionArg"

printf "\nReady to deploy small Universal Sentence Encoder for similarity server \n\n"
