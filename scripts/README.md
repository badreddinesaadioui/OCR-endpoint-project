# Scripts Index

## Benchmark and Analysis

1. `export_benchmark_results.py`  
   Export OCR/LLM benchmark DB results into `exports/` CSV/JSON/MD artifacts.

2. `generate_decision_report.py`  
   Generate decision report and charts from exported benchmark artifacts.

## Cloud Run Deployment

1. `cloud_run_deploy.ps1`  
   Build, secret setup, and deploy API to Cloud Run.

2. `cloud_run_validate.ps1`  
   Post-deploy functional validation (`/health`, async job flow).

## Typical Flow

1. Run API locally and validate.
2. Deploy with `cloud_run_deploy.ps1`.
3. Validate deployment with `cloud_run_validate.ps1`.
4. Open PR once docs and smoke tests are green.
