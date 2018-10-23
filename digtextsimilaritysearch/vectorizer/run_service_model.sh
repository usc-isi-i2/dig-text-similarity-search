# Connect to TF Env
source activate dig_text_similarity


# Update Docker TF Serving 
printf "\n"
docker pull tensorflow/serving


# Construct Docker Run Instructions
printf "\n"

PORT=8501

VECTORIZER_NAME="USE-lite-v2"

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
MODEL_DIR="$REPO_DIR/digtextsimilaritysearch/vectorizer/service_models/$VECTORIZER_NAME/"

MOUNT_INSTRUCTIONS="type=bind,source=$MODEL_DIR,target=/models/$VECTORIZER_NAME"


# Run It
docker run -d -p "$PORT:$PORT" --mount "$MOUNT_INSTRUCTIONS" -e "MODEL_NAME=$VECTORIZER_NAME" -t tensorflow/serving -d


# Report Status
printf "\nUsing port $PORT for requests to $VECTORIZER_NAME\n\n"
docker ps
printf "\n"


