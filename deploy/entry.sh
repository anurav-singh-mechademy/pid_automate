#!/bin/sh
# On Lambda AWS_LAMBDA_RUNTIME_API is set; locally, run under the Runtime Interface Emulator if it is mounted.
if [ -z "$AWS_LAMBDA_RUNTIME_API" ] && [ -x /aws-lambda-rie ]; then
  exec /aws-lambda-rie python -m awslambdaric "$@"
fi
exec python -m awslambdaric "$@"
