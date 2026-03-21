# Kubernetes Deployment

This directory contains the Kubernetes manifests for running the DSPy API and Qdrant in a cluster. Use it after you already have a container image and want a minimal production-style deployment with health checks, persistent storage, autoscaling, and ingress.

## Included Manifests

- `deployment.yaml` creates the namespace, API config, API secret placeholder, artifacts PVC, API deployment, service, HPA, and ingress.
- `qdrant.yaml` creates the Qdrant PVC, StatefulSet, and service.

## Prerequisites

- Kubernetes cluster with `kubectl` access
- A published API image that replaces the placeholder image in `deployment.yaml`
- Real values for `OPENAI_API_KEY` and, if needed, `ANTHROPIC_API_KEY`
- A storage class named `standard`, or edits to match your cluster
- An ingress controller if you plan to use the included ingress resource

## Deployment Flow

1. Build and push the API image.

```bash
docker build -t ghcr.io/your-org/dspy-api:latest .
docker push ghcr.io/your-org/dspy-api:latest
```

2. Update placeholders in `deployment.yaml`.

Replace:

- `ghcr.io/your-org/dspy-api:latest`
- API keys in `stringData`
- ingress hostnames and TLS secret names if you will expose the service publicly

3. Apply the API manifests.

```bash
kubectl apply -f k8s/deployment.yaml
```

4. Deploy Qdrant.

```bash
kubectl apply -f k8s/qdrant.yaml
kubectl wait --for=condition=ready pod -l app=qdrant -n dspy-production --timeout=300s
```

5. Verify the rollout.

```bash
kubectl get all -n dspy-production
kubectl logs -f deployment/dspy-api -n dspy-production
kubectl port-forward service/dspy-api-service 8000:80 -n dspy-production
```

Open `http://localhost:8000/docs` after port-forwarding.

## Configuration

The API deployment reads runtime settings from the `dspy-config` ConfigMap in `deployment.yaml`.

Relevant keys:

- `STUDENT_MODEL`
- `ENVIRONMENT`
- `LOG_LEVEL`
- `QDRANT_HOST`
- `QDRANT_PORT`

The deployment also mounts `/app/artifacts` from the `dspy-artifacts-pvc` claim. If you want optimized programs in the cluster, you need to place compiled artifacts there.

## Updating Artifacts

One direct approach is to copy a compiled artifact into a running pod and restart the deployment:

```bash
kubectl cp artifacts/compiled_programs/rag_v1_mipro.json \
  dspy-production/$(kubectl get pod -n dspy-production -l app=dspy-api -o jsonpath='{.items[0].metadata.name}'):/app/artifacts/compiled_programs/

kubectl rollout restart deployment/dspy-api -n dspy-production
```

Use a more durable artifact delivery path if you plan to operate this in production.

## Scaling And Access

Manual scaling:

```bash
kubectl scale deployment dspy-api --replicas=5 -n dspy-production
```

The included HPA targets CPU and memory and scales between 3 and 10 replicas.

The included ingress expects an NGINX-style ingress class and placeholder hostnames. Adjust those settings before exposing the service publicly.

## Troubleshooting

Pods not becoming ready:

```bash
kubectl describe pod -n dspy-production -l app=dspy-api
kubectl get events -n dspy-production --sort-by='.lastTimestamp'
```

Configuration or secret issues:

```bash
kubectl get configmap dspy-config -n dspy-production -o yaml
kubectl get secret dspy-secrets -n dspy-production -o yaml
```

Service reachable only inside the cluster:

- confirm the `Service` exists
- use `kubectl port-forward` for local validation
- check ingress controller status if external traffic is expected

## Related Docs

- [Main README](../README.md)
- [Quick Start](../QUICKSTART.md)
