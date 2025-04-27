
# Machine Learning API Service (FastAPI)

This project provides the machine learning backend for solar energy forecasting. It is built using FastAPI and serves predictive models via RESTful endpoints.

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/entl/evolyte-ml-backend
cd evolyte-ml-backend
```

### 2. Create a virtual environment and activate it

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install required dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the FastAPI ML service

```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Access the API documentation

Once the server is running:

- Open **Swagger UI** (interactive API docs):  
  [http://localhost:8000/docs](http://localhost:8000/docs)

- Open **ReDoc** documentation (alternative style):  
  [http://localhost:8000/redoc](http://localhost:8000/redoc)
