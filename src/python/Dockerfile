FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY src/python/requirements.txt /app/src/python/requirements.txt
RUN pip install --no-cache-dir -r /app/src/python/requirements.txt

# Copy the code (the data is mounted at runtime, not baked into the image)
COPY src/python/ /app/src/python/

CMD ["python", "src/python/titanic.py"]
