# Phase 5: Rollout and Observability Runbook

## Goal

Deploy Python 3.14 image tags progressively, verify runtime behavior under production traffic, and keep a fast rollback path to Python 3.12 tags.

## Inputs

- Helm chart path: `k8s/welearn-api`
- Deployment name: `welearn-api`
- Health endpoint: `/health/`
- Image tag source: `image.tag: {{ .Values.application.revision }}` from `k8s/welearn-api/values.yaml`

## Required Artifacts

Before rollout, ensure both image tags are pushed for the same code revision:

- `<revision>-py314`
- `<revision>-py312`

Example (see Phase 4 document for build details):

```bash
docker tag welearn-api:py314 <registry>/welearn-api:<revision>-py314
docker tag welearn-api:py312 <registry>/welearn-api:<revision>-py312
docker push <registry>/welearn-api:<revision>-py314
docker push <registry>/welearn-api:<revision>-py312
```

## Staging Rollout

1. Deploy Python 3.14 tag to staging:

```bash
helm upgrade --install welearn-api k8s/welearn-api \
  -f k8s/welearn-api/values.staging.yaml \
  --set application.revision=<revision>-py314
```

2. Wait for rollout:

```bash
kubectl rollout status deployment/welearn-api -n <staging-namespace> --timeout=5m
```

3. Verify live pod image:

```bash
kubectl get pods -n <staging-namespace> -l app.kubernetes.io/name=welearn-api \
  -o jsonpath='{range .items[*]}{.metadata.name}{" -> "}{.spec.containers[0].image}{"\n"}{end}'
```

4. Health probe spot-check:

```bash
kubectl get deploy welearn-api -n <staging-namespace>
kubectl get events -n <staging-namespace> --sort-by=.metadata.creationTimestamp | tail -n 30
```

## Production Rollout

1. Deploy Python 3.14 tag to production:

```bash
helm upgrade --install welearn-api k8s/welearn-api \
  -f k8s/welearn-api/values.prod.yaml \
  --set application.revision=<revision>-py314
```

2. Confirm rollout completes:

```bash
kubectl rollout status deployment/welearn-api -n <prod-namespace> --timeout=5m
```

3. Confirm readiness and restarts remain stable:

```bash
kubectl get pods -n <prod-namespace> -l app.kubernetes.io/name=welearn-api
kubectl describe deployment welearn-api -n <prod-namespace>
```

## Monitoring Window (Recommended 30-60 min)

Track these signals during and after rollout:

- API health success rate on `/health/`
- 5xx error rate (global and per endpoint)
- P95/P99 latency for:
  - `/api/v1/search/*`
  - `/api/v1/qna/*`
- Pod restarts and OOM kills
- CPU/memory drift versus Python 3.12 baseline

Suggested pass criteria:

- No sustained increase in 5xx rate
- No crash loops / no unusual restart burst
- P95 latency regression < 10% versus baseline for critical endpoints

## Rollback Decision Rules

Rollback immediately if any of the following persists for more than 10 minutes:

- Elevated 5xx rate relative to baseline
- Repeated pod restarts or failed readiness
- Material latency regression affecting user paths

## Rollback Execution

Option A: Explicit tag rollback to Python 3.12 image

```bash
helm upgrade --install welearn-api k8s/welearn-api \
  -f k8s/welearn-api/values.prod.yaml \
  --set application.revision=<revision>-py312
kubectl rollout status deployment/welearn-api -n <prod-namespace> --timeout=5m
```

Option B: Helm history rollback

```bash
helm history welearn-api -n <prod-namespace>
helm rollback welearn-api <previous-revision> -n <prod-namespace>
kubectl rollout status deployment/welearn-api -n <prod-namespace> --timeout=5m
```

## Post-Rollout Record

Capture in migration notes:

- Deployed revision and image tag
- Rollout start/end timestamps
- Key metrics snapshot during monitoring window
- Rollback performed or not performed
- Any follow-up action items
