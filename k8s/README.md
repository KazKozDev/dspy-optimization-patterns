# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying DSPy in production.

## Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Helm (optional, for cert-manager)
- Container registry access

## Quick Start

### 1. Create Namespace

```bash
kubectl apply -f deployment.yaml
```

This creates the `dspy-production` namespace and all resources.

### 2. Configure Secrets

**Important**: Replace placeholder secrets with real values!

```bash
# Create secret from file
kubectl create secret generic dspy-secrets \
  --from-literal=OPENAI_API_KEY=sk-your-key \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-your-key \
  -n dspy-production

# Or edit the secret in deployment.yaml before applying
```

### 3. Deploy Qdrant Vector DB

```bash
kubectl apply -f qdrant.yaml
```

Wait for Qdrant to be ready:

```bash
kubectl wait --for=condition=ready pod -l app=qdrant -n dspy-production --timeout=300s
```

### 4. Build and Push Docker Image

```bash
# Build
docker build -t ghcr.io/your-org/dspy-api:v1.0.0 .

# Push
docker push ghcr.io/your-org/dspy-api:v1.0.0
```

### 5. Deploy API

Update `deployment.yaml` with your image name, then:

```bash
kubectl apply -f deployment.yaml
```

### 6. Verify Deployment

```bash
# Check pods
kubectl get pods -n dspy-production

# Check logs
kubectl logs -f deployment/dspy-api -n dspy-production

# Port forward for testing
kubectl port-forward service/dspy-api-service 8000:80 -n dspy-production
```

Visit: http://localhost:8000/docs

## Configuration

### Environment Variables

Edit `ConfigMap` in `deployment.yaml`:

```yaml
data:
  STUDENT_MODEL: "gpt-4o-mini"  # Change model
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
```

### Scaling

#### Manual Scaling

```bash
kubectl scale deployment dspy-api --replicas=5 -n dspy-production
```

#### Auto-scaling

The HPA is configured to scale between 3-10 replicas based on CPU/memory.

To adjust:

```yaml
spec:
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

### Resource Limits

Adjust based on your workload:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

## Ingress Setup

### Install Ingress Controller

```bash
# NGINX Ingress
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml
```

### Install Cert-Manager (for TLS)

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### Configure Domain

Update `deployment.yaml` Ingress section:

```yaml
spec:
  tls:
  - hosts:
    - api.yourdomain.com  # Your domain
    secretName: dspy-api-tls
  rules:
  - host: api.yourdomain.com
```

## Updating Compiled Programs

Compiled programs are stored in PVC. To update:

### Option 1: Direct Upload

```bash
# Copy artifact to pod
kubectl cp artifacts/compiled_programs/rag_v2.json \
  dspy-production/dspy-api-xxxxx:/app/artifacts/compiled_programs/

# Restart pods to load new artifact
kubectl rollout restart deployment/dspy-api -n dspy-production
```

### Option 2: CI/CD Pipeline

See `.github/workflows/ci.yml` for automated deployment on git push.

## Monitoring

### View Logs

```bash
# All pods
kubectl logs -f -l app=dspy-api -n dspy-production

# Specific pod
kubectl logs -f dspy-api-xxxxx -n dspy-production

# Previous container (if crashed)
kubectl logs dspy-api-xxxxx -n dspy-production --previous
```

### Metrics

Install Prometheus + Grafana:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
```

### Phoenix Observability

Deploy Phoenix for request tracing:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: phoenix
  namespace: dspy-production
spec:
  template:
    spec:
      containers:
      - name: phoenix
        image: arizephoenix/phoenix:latest
        ports:
        - containerPort: 6006
```

## Troubleshooting

### Pods Not Starting

```bash
# Describe pod
kubectl describe pod dspy-api-xxxxx -n dspy-production

# Check events
kubectl get events -n dspy-production --sort-by='.lastTimestamp'
```

### Out of Memory

Increase memory limits in `deployment.yaml`.

### High Latency

1. Check HPA scaling: `kubectl get hpa -n dspy-production`
2. Increase replicas: `kubectl scale deployment dspy-api --replicas=10`
3. Review metrics: Are pods CPU/memory saturated?

### API Key Issues

Verify secrets:

```bash
kubectl get secret dspy-secrets -n dspy-production -o yaml
```

## Backup and Disaster Recovery

### Backup Artifacts

```bash
# Create backup of PVC
kubectl exec -n dspy-production dspy-api-xxxxx -- tar czf /tmp/artifacts-backup.tar.gz /app/artifacts

kubectl cp dspy-production/dspy-api-xxxxx:/tmp/artifacts-backup.tar.gz ./backup.tar.gz
```

### Backup Qdrant Data

```bash
# Snapshot Qdrant
kubectl exec -n dspy-production qdrant-0 -- curl -X POST http://localhost:6333/collections/documents/snapshots

# Download snapshot (see Qdrant docs)
```

## Production Checklist

- [ ] Configure real secrets (not placeholders)
- [ ] Set up domain and TLS certificates
- [ ] Configure resource limits based on load testing
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log aggregation (ELK/Loki)
- [ ] Set up alerting (PagerDuty/Opsgenie)
- [ ] Test disaster recovery procedures
- [ ] Document runbooks for common issues
- [ ] Set up cost monitoring
- [ ] Configure network policies for security
- [ ] Enable Pod Security Policies
- [ ] Set up backup automation

## Cost Optimization

### Use Spot Instances

For non-critical workloads:

```yaml
spec:
  template:
    spec:
      nodeSelector:
        node.kubernetes.io/instance-type: spot
      tolerations:
      - key: "spot"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
```

### Right-size Resources

Monitor actual usage and adjust requests/limits to avoid over-provisioning.

### Use Cheaper Models

For staging/dev environments, use cheaper models in ConfigMap.
