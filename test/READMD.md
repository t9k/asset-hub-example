# README

## Train

部署 Workflow Template：

```bash
(
    cd base
    kubectl apply -n demo -f apply-workflow.yaml -f download-asset.yaml -f git-clone.yaml -f launcher-retrain.yaml -f upload-model.yaml
)
```

创建 pvc

```bash
kubectl apply -f pvc.yaml -n demo
```

创建 run

```bash
kubectl apply -f run.yaml -n demo
```
