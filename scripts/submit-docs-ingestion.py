#!/usr/bin/env python3
"""Upload fixed docs RAG pipeline and submit Milvus ingestion run."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

from kfp import Client

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_YAML = ROOT / "pipelines" / "github_rag_pipeline.yaml"
KFP_HOST = os.environ.get("KFP_HOST", "http://127.0.0.1:8888")
KFP_NAMESPACE = os.environ.get("KFP_NAMESPACE", "user")
KFP_USER = os.environ.get("KFP_USER", "user@example.com")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "code-rag-test")
PIPELINE_ID = os.environ.get("PIPELINE_ID", "8410d5da-9380-497d-be74-866bee8024df")

CONTEXT_PATH = Path.home() / ".config" / "kfp" / "context.json"


def main() -> None:
    CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTEXT_PATH.write_text(
        json.dumps(
            {
                "namespace": KFP_NAMESPACE,
                "client_authentication_header_name": "kubeflow-userid",
                "client_authentication_header_value": KFP_USER,
            }
        )
    )

    print("==> Recompiling pipeline")
    import subprocess

    subprocess.run(["python3", "kubeflow-pipeline.py"], cwd=ROOT / "pipelines", check=True)

    stamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M")
    client = Client(host=KFP_HOST, namespace=KFP_NAMESPACE)

    print("==> Uploading pipeline version")
    version = client.upload_pipeline_version(
        pipeline_package_path=str(PIPELINE_YAML),
        pipeline_version_name=f"docs-md-fixed-images-{stamp}",
        pipeline_id=PIPELINE_ID,
        description="Fixed docker.io image names for OKE short-name enforcement",
    )
    print("pipeline_version_id:", version.pipeline_version_id)

    print("==> Submitting pipeline run")
    result = client.create_run_from_pipeline_package(
        pipeline_file=str(PIPELINE_YAML),
        run_name=f"docs-ingestion-fixed-{stamp}",
        experiment_name=EXPERIMENT_NAME,
        namespace=KFP_NAMESPACE,
        enable_caching=False,
        arguments={
            "repo_owner": "kubeflow",
            "repo_name": "website",
            "directory_path": "content/en/docs",
            "github_token": "",
            "base_url": "https://www.kubeflow.org/docs",
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "milvus_host": "my-release-milvus.docs-agent.svc.cluster.local",
            "milvus_port": "19530",
            "collection_name": "kubeflow_docs_docs_rag",
        },
    )
    print("run_id:", result.run_id)
    print("run_details:", result)


if __name__ == "__main__":
    main()
