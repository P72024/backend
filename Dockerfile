
# Use the official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app files
COPY . .

# Expose the backend port
EXPOSE 5000

# Command to run the backend
CMD ["python", "app/main.py"]
