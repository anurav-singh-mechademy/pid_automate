#!/usr/bin/env bash
# Push the built image to the account's ECR repo <NAME>, creating the repo the first time. Tags are immutable: a tag,
# once pushed, always means the same image. Reads deploy/config.env.
#   deploy/push.sh
. "$(dirname "$0")/common.sh"
need AWS_PROFILE AWS_REGION NAME WEIGHTS_SHA256
TAG=$(image_tag)
docker image inspect "$NAME:$TAG" >/dev/null 2>&1 || { echo "no local image $NAME:$TAG: run deploy/build.sh first" >&2; exit 1; }
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com

aws ecr describe-repositories --repository-names "$NAME" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$NAME" --image-scanning-configuration scanOnPush=true \
       --image-tag-mutability IMMUTABLE >/dev/null
if aws ecr describe-images --repository-name "$NAME" --image-ids imageTag="$TAG" >/dev/null 2>&1; then
  echo "$REGISTRY/$NAME:$TAG is already in ECR"
  exit 0
fi
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
docker tag "$NAME:$TAG" "$REGISTRY/$NAME:$TAG"
docker push "$REGISTRY/$NAME:$TAG"
echo "pushed $REGISTRY/$NAME:$TAG"
