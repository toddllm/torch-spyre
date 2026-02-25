# Code Atlas CloudFront Deploy

## Bootstrap (one-time)

```bash
cd /Users/tdeshane/torch-spyre-open-work/code-atlas
./deployment/cloudfront/bootstrap_cloudfront_site.sh
```

This creates:

- S3 bucket for static files
- CloudFront distribution (HTTPS URL)
- `deployment/cloudfront/site.env` config

## Deploy (repeatable)

```bash
cd /Users/tdeshane/torch-spyre-open-work/code-atlas
./deployment/cloudfront/deploy_cloudfront_site.sh
```

Deploy does:

- syncs `web/` to S3
- creates CloudFront invalidation for `/*`
- waits for invalidation completion
- smoke-tests the live URL for marker text

## Useful flags

- `--no-wait`: do not wait for invalidation completion
- `--expected-text "<text>"`: override smoke-test marker
- `--source-dir /path/to/web`: override source directory
- `--config /path/to/site.env`: alternate config path
