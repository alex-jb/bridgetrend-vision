# AWS judge deployment

This deployment runs the OpenCV 5 judge console on AWS App Runner, stores full
session records and human reviews in DynamoDB, and publishes latency, safe-action,
trace-validity, and review metrics to CloudWatch.

## Build and publish

~~~bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export ECR_REPOSITORY=bridgetrend-vision

aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" \
  >/dev/null 2>&1 || aws ecr create-repository --repository-name "$ECR_REPOSITORY"

aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin \
  "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker build --platform linux/amd64 -t "$ECR_REPOSITORY:judge-v1" .
docker tag "$ECR_REPOSITORY:judge-v1" \
  "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:judge-v1"
docker push \
  "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:judge-v1"
~~~

## Deploy

~~~bash
aws cloudformation deploy \
  --template-file deploy/aws/apprunner.yaml \
  --stack-name bridgetrend-vision-judge \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ImageUri="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:judge-v1"
~~~

After deployment, obtain the hostname from the `ServiceUrl` stack output and
verify `/healthz`, one single-case run, one human review, the DynamoDB record,
and the CloudWatch dashboard. Pin the final submission image by digest and set
`AutoDeploymentsEnabled` to `false` so the judge build cannot drift.

## Responsible-operation boundary

- The deployed six-case fixture pack is deterministic, CC0, and market-neutral.
- It demonstrates OpenCV 5 perception, bounded evidence acquisition, abstention,
  human control, trace integrity, latency, and failure handling.
- It does not validate U.S./China product transfer or demand forecasting.
- Full OpenCLIP and G1A experiments remain reproducible offline jobs; the small
  judge container uses frozen retrieval inputs to make the OpenCV contribution
  fast and deterministic.
