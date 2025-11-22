# Use a slim Python image for a smaller container size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts and application code
# NOTE: You must run model_trainer.py locally to generate sales_pipeline_model.pkl
COPY sales_pipeline_model.pkl .
COPY api.py .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application using Uvicorn
# 'api:app' refers to the 'app' object inside 'api.py'
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]