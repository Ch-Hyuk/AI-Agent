# AI Agent Application

This repository outlines a simple architecture for a mobile app that leverages free APIs to gather large amounts of data. The collected data is used to provide recommendation and analytics services.

## Directory structure

```
frontend/  - Flutter application targeting Android and iOS
backend/   - FastAPI service for APIs and analytics
database/  - Database schema and initialization scripts
bigdata_pipeline.py - Example Dask-based analytics pipeline
```

## Setup

1. Follow the instructions in `frontend/README.md` to bootstrap the Flutter app.
2. Follow `backend/README.md` for running the FastAPI server.
3. Database related notes are available in `database/README.md`.

The existing `bigdata_pipeline.py` script demonstrates how you can process large datasets and visualize results using Dask and dask-ml.
