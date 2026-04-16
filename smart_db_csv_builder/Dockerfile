FROM python:3.12-slim

WORKDIR /app

# System deps for pyodbc (MSSQL) + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl unixodbc unixodbc-dev gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
