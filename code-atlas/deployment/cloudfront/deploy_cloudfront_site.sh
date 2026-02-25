#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  deploy_cloudfront_site.sh [options]

Options:
  --config <path>        Env config path (default: deployment/cloudfront/site.env)
  --source-dir <path>    Source directory to publish (overrides SOURCE_DIR in config)
  --expected-text <txt>  Smoke marker text (overrides EXPECTED_TEXT in config)
  --no-wait              Do not wait for invalidation completion
  -h, --help             Show this help

Required config vars:
  BUCKET_NAME
  DISTRIBUTION_ID
  CLOUDFRONT_DOMAIN

Optional config vars:
  SOURCE_DIR
  EXPECTED_TEXT
  AWS_REGION (default us-east-1)
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="$ROOT_DIR/deployment/cloudfront/site.env"
WAIT_FOR_INVALIDATION="true"
SOURCE_DIR_OVERRIDE=""
EXPECTED_TEXT_OVERRIDE=""
SMOKE_ATTEMPTS="${SMOKE_ATTEMPTS:-20}"
SMOKE_RETRY_SECONDS="${SMOKE_RETRY_SECONDS:-15}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --source-dir)
      SOURCE_DIR_OVERRIDE="$2"
      shift 2
      ;;
    --expected-text)
      EXPECTED_TEXT_OVERRIDE="$2"
      shift 2
      ;;
    --no-wait)
      WAIT_FOR_INVALIDATION="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

AWS_REGION="${AWS_REGION:-us-east-1}"
SOURCE_DIR="${SOURCE_DIR:-$ROOT_DIR/web}"
EXPECTED_TEXT="${EXPECTED_TEXT:-vLLM-Spyre Integration Atlas}"
S3_WEBSITE_URL="${S3_WEBSITE_URL:-http://${BUCKET_NAME:-invalid}.s3-website-${AWS_REGION}.amazonaws.com}"

if [[ -n "$SOURCE_DIR_OVERRIDE" ]]; then
  SOURCE_DIR="$SOURCE_DIR_OVERRIDE"
fi

if [[ -n "$EXPECTED_TEXT_OVERRIDE" ]]; then
  EXPECTED_TEXT="$EXPECTED_TEXT_OVERRIDE"
fi

if [[ -z "${BUCKET_NAME:-}" || -z "${DISTRIBUTION_ID:-}" || -z "${CLOUDFRONT_DOMAIN:-}" ]]; then
  echo "Config missing BUCKET_NAME, DISTRIBUTION_ID, or CLOUDFRONT_DOMAIN" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_DIR" || ! -f "$SOURCE_DIR/index.html" ]]; then
  echo "Source directory must exist and include index.html: $SOURCE_DIR" >&2
  exit 1
fi

BUILD_DIR="$(mktemp -d "/tmp/code-atlas-site.XXXXXX")"
cleanup() {
  rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

cp -R "$SOURCE_DIR"/. "$BUILD_DIR"/

echo "Uploading site to s3://$BUCKET_NAME"
aws s3 sync "$BUILD_DIR/" "s3://$BUCKET_NAME/" \
  --delete \
  --cache-control "public,max-age=60" \
  --region "$AWS_REGION"

echo "Creating CloudFront invalidation..."
INVALIDATION_ID="$(aws cloudfront create-invalidation \
  --distribution-id "$DISTRIBUTION_ID" \
  --paths "/*" \
  --query 'Invalidation.Id' \
  --output text)"

echo "Invalidation ID: $INVALIDATION_ID"

if [[ "$WAIT_FOR_INVALIDATION" == "true" ]]; then
  echo "Waiting for invalidation completion..."
  aws cloudfront wait invalidation-completed \
    --distribution-id "$DISTRIBUTION_ID" \
    --id "$INVALIDATION_ID"
fi

SITE_URL="https://$CLOUDFRONT_DOMAIN"
TMP_BODY="$(mktemp "/tmp/code-atlas-smoke.XXXXXX")"
trap 'cleanup; rm -f "$TMP_BODY"' EXIT

DIST_STATUS="$(aws cloudfront get-distribution \
  --id "$DISTRIBUTION_ID" \
  --query 'Distribution.Status' \
  --output text 2>/dev/null || echo "Unknown")"

if [[ "$DIST_STATUS" != "Deployed" ]]; then
  echo "Distribution status is '$DIST_STATUS'. Continuing with retrying smoke tests during propagation..."
fi

echo "Running smoke tests against $SITE_URL"
SMOKE_OK="false"
LAST_CURL_EXIT=0
LAST_HTTP_CODE="000"

for (( attempt=1; attempt<=SMOKE_ATTEMPTS; attempt++ )); do
  LAST_HTTP_CODE="000"
  set +e
  LAST_HTTP_CODE="$(curl -sS -o "$TMP_BODY" -w "%{http_code}" "$SITE_URL/")"
  LAST_CURL_EXIT=$?
  set -e

  if [[ "$LAST_CURL_EXIT" -eq 0 && "$LAST_HTTP_CODE" == "200" ]]; then
    if rg -q "$EXPECTED_TEXT" "$TMP_BODY"; then
      SMOKE_OK="true"
      break
    else
      echo "Smoke attempt $attempt/$SMOKE_ATTEMPTS: HTTP 200 but marker text not found yet."
    fi
  else
    echo "Smoke attempt $attempt/$SMOKE_ATTEMPTS: curl_exit=$LAST_CURL_EXIT http=$LAST_HTTP_CODE."
    if [[ "$LAST_CURL_EXIT" -eq 6 && "$DIST_STATUS" != "Deployed" ]]; then
      break
    fi
  fi

  if [[ "$attempt" -lt "$SMOKE_ATTEMPTS" ]]; then
    sleep "$SMOKE_RETRY_SECONDS"
  fi
done

if [[ "$SMOKE_OK" != "true" ]]; then
  if [[ "$LAST_CURL_EXIT" -eq 6 ]]; then
    echo "CloudFront DNS not yet resolvable from this network."
    echo "Temporary fallback URL: $S3_WEBSITE_URL"
    echo "Running fallback smoke test..."
    FALLBACK_BODY="$(mktemp "/tmp/code-atlas-fallback-smoke.XXXXXX")"
    trap 'cleanup; rm -f "$TMP_BODY" "$FALLBACK_BODY"' EXIT
    FALLBACK_HTTP_CODE="$(curl -sS -o "$FALLBACK_BODY" -w "%{http_code}" "$S3_WEBSITE_URL/")"
    if [[ "$FALLBACK_HTTP_CODE" == "200" ]] && rg -q "$EXPECTED_TEXT" "$FALLBACK_BODY"; then
      echo "Fallback smoke test passed (HTTP 200 + marker text)."
      echo "Deploy completed; CloudFront is still propagating."
      echo "CloudFront URL: $SITE_URL"
      echo "S3 fallback URL: $S3_WEBSITE_URL"
      echo "Invalidation ID: $INVALIDATION_ID"
      exit 0
    fi
  fi
  echo "Smoke test failed after $SMOKE_ATTEMPTS attempts (last curl_exit=$LAST_CURL_EXIT http=$LAST_HTTP_CODE)." >&2
  exit 1
fi

echo "Deploy successful."
echo "Live URL: $SITE_URL"
echo "S3 fallback URL: $S3_WEBSITE_URL"
echo "Invalidation ID: $INVALIDATION_ID"
echo "Verified marker text: $EXPECTED_TEXT"
