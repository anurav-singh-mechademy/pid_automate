#!/usr/bin/env bash
# Deploy the two P&ID model APIs (<NAME>-detect, <NAME>-ocr) from an image already pushed with push.sh, as Lambda
# functions with IAM-auth function URLs, plus their input bucket and role. Safe to re-run: creates what is missing,
# updates the rest. Reads deploy/config.env.
#   deploy/deploy.sh                       deploy the image for the current commit
#   IMAGE_TAG=<tag> deploy/deploy.sh       deploy (or roll back to) any tag already in ECR
# The first run needs iam:CreateRole and iam:PassRole; if the role <NAME>-lambda-role already exists it is left as is.
. "$(dirname "$0")/common.sh"
need AWS_PROFILE AWS_REGION NAME WEIGHTS_SHA256
CALLER_ROLE_ARN=${CALLER_ROLE_ARN:-}   # optional; empty = same-account users only
PREFIX=pages/                 # the only place the functions may read pages from
EXPIRE_DAYS=7                 # pages are client drawings: delete them after a week
MEMORY_MB=10240               # ~6 vCPUs; threads follow the CPUs Lambda gives
DETECT_TIMEOUT=300
OCR_TIMEOUT=900               # the maximum; billed only for time used

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=$AWS_REGION
BUCKET=$NAME-input-$ACCOUNT
ROLE=$NAME-lambda-role
TAG=${IMAGE_TAG:-$(image_tag)}
IMAGE_URI=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$NAME:$TAG
aws ecr describe-images --repository-name "$NAME" --image-ids imageTag="$TAG" >/dev/null 2>&1 \
  || { echo "$IMAGE_URI is not in ECR: run deploy/build.sh and deploy/push.sh first" >&2; exit 1; }
echo "deploying $IMAGE_URI"

# ---------- input bucket: private, encrypted, TLS only, pages expire, only the caller role writes ----------
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
fi
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration \
  "{\"Rules\":[{\"ID\":\"expire-pages\",\"Status\":\"Enabled\",\"Filter\":{\"Prefix\":\"$PREFIX\"},\"Expiration\":{\"Days\":$EXPIRE_DAYS}}]}"
aws s3api put-bucket-policy --bucket "$BUCKET" --policy "$(cat <<EOF
{"Version":"2012-10-17","Statement":[
 {"Sid":"TlsOnly","Effect":"Deny","Principal":"*","Action":"s3:*",
  "Resource":["arn:aws:s3:::$BUCKET","arn:aws:s3:::$BUCKET/*"],"Condition":{"Bool":{"aws:SecureTransport":"false"}}},
 {"Sid":"CallerWritesPages","Effect":"Allow","Principal":{"AWS":"${CALLER_ROLE_ARN:-arn:aws:iam::$ACCOUNT:root}"},
  "Action":["s3:PutObject","s3:GetObject","s3:DeleteObject"],"Resource":"arn:aws:s3:::$BUCKET/$PREFIX*"}]}
EOF
)"

# ---------- the functions' role: read pages, write logs (left as is if it exists, e.g. made by hand) ----------
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --description "P&ID model APIs: read input pages, write logs" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  aws iam put-role-policy --role-name "$ROLE" --policy-name read-pages --policy-document \
    "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"s3:GetObject\",\"Resource\":\"arn:aws:s3:::$BUCKET/$PREFIX*\"}]}"
  NEW_ROLE=1
fi
ROLE_ARN=$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)
[ -z "${NEW_ROLE:-}" ] || sleep 10   # a new role takes a few seconds before Lambda can assume it

# ---------- functions + IAM-auth URLs, callable by the caller role only ----------
deploy_fn() {   # name handler timeout
  local fn=$NAME-$1 env="Variables={PID_INPUT_BUCKET=$BUCKET,PID_INPUT_PREFIX=$PREFIX}"
  if aws lambda get-function --function-name "$fn" >/dev/null 2>&1; then
    aws lambda update-function-code --function-name "$fn" --image-uri "$IMAGE_URI" >/dev/null
    aws lambda wait function-updated-v2 --function-name "$fn"
    aws lambda update-function-configuration --function-name "$fn" --role "$ROLE_ARN" --memory-size $MEMORY_MB \
      --timeout "$3" --ephemeral-storage Size=2048 --environment "$env" --image-config "Command=$2" >/dev/null
  else
    aws lambda create-function --function-name "$fn" --package-type Image --code ImageUri="$IMAGE_URI" \
      --architectures arm64 --role "$ROLE_ARN" --memory-size $MEMORY_MB --timeout "$3" --ephemeral-storage Size=2048 \
      --environment "$env" --image-config "Command=$2" \
      --description "P&ID model API ($1), image $TAG" >/dev/null
    aws logs create-log-group --log-group-name "/aws/lambda/$fn" 2>/dev/null || true
    aws logs put-retention-policy --log-group-name "/aws/lambda/$fn" --retention-in-days 30
  fi
  aws lambda wait function-updated-v2 --function-name "$fn"
  aws lambda get-function-url-config --function-name "$fn" >/dev/null 2>&1 \
    || aws lambda create-function-url-config --function-name "$fn" --auth-type AWS_IAM >/dev/null
  # a function URL call needs both permissions; with no caller role, only the account's own users can call
  for action in lambda:InvokeFunctionUrl lambda:InvokeFunction; do
    sid=caller-${action#lambda:}
    aws lambda remove-permission --function-name "$fn" --statement-id "$sid" >/dev/null 2>&1 || true
    [ -n "$CALLER_ROLE_ARN" ] || continue
    if [ "$action" = lambda:InvokeFunctionUrl ]; then
      aws lambda add-permission --function-name "$fn" --statement-id "$sid" --action "$action" \
        --principal "$CALLER_ROLE_ARN" --function-url-auth-type AWS_IAM >/dev/null
    else
      aws lambda add-permission --function-name "$fn" --statement-id "$sid" --action "$action" \
        --principal "$CALLER_ROLE_ARN" >/dev/null
    fi
  done
  echo "$fn: $(aws lambda get-function-url-config --function-name "$fn" --query FunctionUrl --output text)"
}
deploy_fn detect handler.detect $DETECT_TIMEOUT
deploy_fn ocr handler.ocr $OCR_TIMEOUT
echo "pages go to s3://$BUCKET/$PREFIX (deleted after $EXPIRE_DAYS days)"
