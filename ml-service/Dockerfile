FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (all service modules, including the DRE)
COPY *.py ./

# Model artifacts: model_info.json and gcn_info.json are in the repository.
# The weights (bilstm.pt, gcn.pt) and pickles (tokenizer.pkl, label_encoder.pkl)
# are not published and must be placed in artifacts/ before building.
COPY artifacts/ artifacts/

EXPOSE 5000

# Secrets (MONGODB_URI, ANTHROPIC_API_KEY) are passed at runtime with --env-file.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "30", "app:app"]
