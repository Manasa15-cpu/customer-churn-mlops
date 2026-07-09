# Customer Churn Prediction with MLOps

## Project Overview

This project predicts whether a customer is likely to churn using a Machine Learning model built with XGBoost. The application is deployed as a REST API using FastAPI and containerized with Docker for easy deployment.

## Features

- Customer churn prediction using Machine Learning
- REST API built with FastAPI
- Dockerized application
- Model saved using Joblib
- Interactive API testing with Swagger UI
- GitHub version control

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Uvicorn
- Docker
- Git
- GitHub

## Project Structure


customer-churn-mlops/
│
├── data/
├── models/
├── app.py
├── predict.py
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore


## How to Run

### Clone the repository

bash
git clone https://github.com/Manasa15-cpu/customer-churn-mlops.git
cd customer-churn-mlops


### Build Docker Image

bash
docker build -t customer-churn-app .


### Run Docker Container

bash
docker run -p 5000:5000 customer-churn-app


### Open Swagger UI


http://localhost:5000/docs


## API Endpoint

*POST* /predict

Example Request

json
{
  "SeniorCitizen": 1,
  "tenure": 12,
  "MonthlyCharges": 95,
  "TotalCharges": 1140
}


Example Response

json
{
  "Prediction": "No Churn"
}


## Future Improvements

- Model monitoring
- MLflow integration
- Kubernetes deployment
- CI/CD pipeline using GitHub Actions
- Cloud deployment on AWS or Azure

## Author

*Manasa Degam*
