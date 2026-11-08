# Dashboard image. The data and model are NOT baked in (they are not in the repository):
# the first run downloads the dataset and trains the model, which takes about 2 minutes.
#
#   docker build -t churn-dashboard .
#   docker run --rm -p 8501:8501 -v churn-artifacts:/app/models -v churn-data:/app/data \
#       churn-dashboard
#
# Then open http://localhost:8501. The named volumes keep the model between runs.
FROM python:3.14-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# libgomp is needed by XGBoost and LightGBM.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY . .

EXPOSE 8501

# .streamlit/config.toml binds to localhost for local use; inside a container the server must
# listen on all interfaces, so the address is overridden here. Publish the port only to a
# trusted network (for example -p 127.0.0.1:8501:8501).
CMD ["sh", "-c", "[ -f models/calibrated_model.joblib ] || (python -m src.data.download && python -m src.pipeline); exec python -m streamlit run app/app.py --server.address 0.0.0.0"]
