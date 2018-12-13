#!/usr/bin/env bash
printf "\nOptional Arg: model version number (highest number will be deployed) \n\n"

# Make Dockerized Model
python dt_sim/vectorizer/make_service_model.py -v "$1"

printf "\nReady to deploy dockerized Universal Sentence Encoder with service_model_run.sh \n\n"

