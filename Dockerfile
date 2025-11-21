# ===========================================
# 1. Base Image
# ===========================================
FROM python:3.10-slim

# -------------------------------------------
# 2. Set working directory inside container
# -------------------------------------------
WORKDIR /app

# -------------------------------------------
# 3. Copy requirements first (for caching)
# -------------------------------------------
COPY requirements.txt .

# Install dependencies (no cache → smaller image)
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------
# 4. Copy entire project into the container
# -------------------------------------------
COPY . .

# -------------------------------------------
# 5. Expose API port
# -------------------------------------------
EXPOSE 8000

# -------------------------------------------
# 6. Start FastAPI app with Uvicorn
# -------------------------------------------
CMD ["uvicorn", "src.api.paysim_api:app", "--host", "0.0.0.0", "--port", "8000"]
