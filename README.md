\# House Price MLOps Pipeline



A production-style machine learning pipeline for predicting California housing prices using Python, scikit-learn, MLflow, automated testing, and reproducible project structure.



This project demonstrates core MLOps skills, including data ingestion, preprocessing, model training, evaluation, experiment tracking, model artifact saving, and automated pipeline testing.



\---



\## Project Overview



The goal of this project is to build a reproducible machine learning workflow that predicts median house values from housing and geographic features.



Rather than building only a notebook-based model, this project is organized as a modular ML pipeline similar to what a production machine learning team would use.



\---



\## Tech Stack



\- Python

\- Pandas

\- NumPy

\- scikit-learn

\- MLflow

\- Joblib

\- Pytest

\- Git / GitHub



\---



\## Project Structure



```text

house-price-mlops-pipeline/

│

├── data/

│   ├── raw/

│   └── processed/

│

├── models/

│

├── reports/

│

├── src/

│   ├── ingest\_data.py

│   ├── preprocess.py

│   ├── train.py

│   └── evaluate.py

│

├── tests/

│   └── test\_data\_pipeline.py

│

├── requirements.txt

├── .gitignore

└── README.md

