# 1. Use a lightweight Python environment
FROM python:3.11-slim

# 2. Set the "Home" directory inside the container
WORKDIR /app

# 3. Copy only the requirements first (this makes builds faster)
COPY requirements.txt .

# 4. Install the libraries inside the container
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your code and the model (.pkl file) into the container
COPY . .

# 6. Tell the container to listen on Port 8000
EXPOSE 8000

# 7. The command that starts the Waiter (API)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]