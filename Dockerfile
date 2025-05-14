FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH="/app"

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy files
COPY ./app ./app
COPY ./output_graph ./output_graph
COPY ./tests ./tests

# Install and run tests
#RUN pytest tests/


# Launch app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
