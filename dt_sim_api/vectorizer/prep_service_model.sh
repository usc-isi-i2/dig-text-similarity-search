# Connect to TF Env
source activate dig_text_similarity 


# Make Docker Runable Model
VERSION="001"
python digtextsimilaritysearch/vectorizer/make_service_model.py -v "$VERSION"

printf "\nReady to deploy dockerized Universal Sentence Encoder with service_model_run.sh \n\n"

