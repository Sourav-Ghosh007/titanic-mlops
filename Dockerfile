# Dockerfile
# This is the "recipe" for building a container image of our Titanic API

# STEP 1: Start from an official, lightweight Python image
# "slim" version = smaller size, faster to download/build
FROM python:3.11-slim

# STEP 2: Set the working directory inside the container
# All following commands run from this folder inside the container
WORKDIR /app

# STEP 3: Copy only requirements.txt first (not all the code yet)
# This is a Docker best practice: if your code changes but dependencies don't,
# Docker reuses the cached install step instead of reinstalling everything - much faster rebuilds
COPY requirements.txt .

# STEP 4: Install dependencies inside the container
RUN pip install --no-cache-dir -r requirements.txt

# STEP 5: Now copy the rest of the project files (app.py, model.pkl)
COPY . .

# STEP 6: Tell Docker which port the app listens on
# This is documentation for humans/tools - it does NOT actually publish the port
EXPOSE 8000

# STEP 7: The command that runs when the container starts
# --host 0.0.0.0 is required inside Docker so the app is reachable from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
