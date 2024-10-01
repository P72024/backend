# Use the python:3.10.12 base image
FROM python:3.10.12

# Argument to set the environment (development/production)
ARG ENV

# Set the working directory inside the container
WORKDIR /app/

# Copy all files to the working directory
COPY . .

RUN mkdir -p /usr/local/cuda

# Set environment variable based on the ARG
ENV mode=$ENV

# Verbose logging to ensure each step is tracked
ENTRYPOINT [ "sh", "./prod.sh" ]

# Entry point to run the Python application
# CMD [ "python3", "./src/" ]