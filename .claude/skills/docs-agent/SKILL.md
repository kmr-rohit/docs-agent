```markdown
# docs-agent Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill provides a comprehensive guide to the development patterns, coding conventions, and common workflows used in the `docs-agent` Python repository. It covers file naming, code style, commit practices, and step-by-step instructions for maintaining pipelines and documentation. The guide is designed for contributors seeking to align with established practices and efficiently manage coordinated changes across pipeline scripts and documentation.

## Coding Conventions

- **File Naming:**  
  Use kebab-case for all file names.
  ```
  # Good
  incremental-pipeline.py
  code-pipeline.py

  # Bad
  IncrementalPipeline.py
  incremental_pipeline.py
  ```

- **Import Style:**  
  Use relative imports within the package.
  ```python
  # Good
  from .utils import helper_function

  # Bad
  import utils.helper_function
  ```

- **Export Style:**  
  Use named exports (define functions/classes explicitly, avoid wildcard exports).
  ```python
  # Good
  def run_pipeline():
      pass

  __all__ = ['run_pipeline']

  # Bad
  from .pipeline import *
  ```

- **Commit Messages:**  
  Follow the conventional commit format with prefixes such as `fix:`, `docs:`, `chore:`, `perf:`.
  ```
  docs: update troubleshooting section for Milvus connectivity
  fix: pin numpy version in all pipeline scripts
  ```

## Workflows

### Pipeline Code Change
**Trigger:** When a change or fix needs to be applied to all or most pipeline scripts for consistency, bugfix, or dependency management.  
**Command:** `/update-pipelines`

1. Identify the change needed across pipelines (e.g., dependency pinning, Docker image reference, Milvus logic).
2. Edit the following files to apply the change:
    - `pipelines/incremental-pipeline.py`
    - `pipelines/issues-pipeline.py`
    - `pipelines/kubeflow-pipeline.py`
    - (sometimes) `pipelines/code-pipeline.py`
3. Update documentation if necessary.
4. Commit all affected pipeline files together with a descriptive message.
    ```
    fix: update Docker image reference in all pipeline scripts
    ```

### Documentation Troubleshooting Update
**Trigger:** When new troubleshooting information, workarounds, or operational tips become available for Kubeflow, Milvus, or related infrastructure.  
**Command:** `/add-troubleshooting-doc`

1. Discover or resolve a new operational issue or workaround.
2. Edit `kube.md` to add a new section or update existing content.
    ```markdown
    ## Milvus Connection Timeout

    If you encounter connection timeouts, ensure the Milvus service is reachable from your cluster nodes...
    ```
3. Commit only `kube.md` with a `docs:` prefix in the message.
    ```
    docs: add troubleshooting for Milvus connection timeouts
    ```

### Pipeline Codegen Recompile
**Trigger:** When the Python pipeline definition (e.g., `kubeflow-pipeline.py`) is updated and the corresponding YAML needs to be recompiled.  
**Command:** `/recompile-pipeline-yaml`

1. Update the Python pipeline definition (e.g., `kubeflow-pipeline.py`).
2. Regenerate the YAML file (e.g., `github_rag_pipeline.yaml`) from the updated Python script.
    ```bash
    # Example command (adjust as needed)
    dsl-compile --py pipelines/kubeflow-pipeline.py --output pipelines/github_rag_pipeline.yaml
    ```
3. Commit the regenerated YAML file with a `chore:` prefix.
    ```
    chore: recompile github_rag_pipeline.yaml after pipeline update
    ```

## Testing Patterns

- **Test File Naming:**  
  Test files follow the pattern `*.test.*` (e.g., `pipeline.test.py`).
- **Framework:**  
  The specific test framework is not detected; use standard Python test frameworks (e.g., `pytest` or `unittest`) as appropriate.
- **Example:**
  ```python
  # pipelines/incremental-pipeline.test.py

  def test_incremental_pipeline_runs():
      assert run_pipeline() is not None
  ```

## Commands

| Command                   | Purpose                                                        |
|---------------------------|----------------------------------------------------------------|
| /update-pipelines         | Apply coordinated changes across all pipeline scripts          |
| /add-troubleshooting-doc  | Add or update troubleshooting documentation in kube.md         |
| /recompile-pipeline-yaml  | Regenerate pipeline YAML from updated Python pipeline script   |
```
