# Use the python:3.10.12 base image
FROM python:3.10.12

# Argument to set the environment (development/production)
ARG ENV

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app/

# Copy all files to the working directory
COPY . .


RUN pip3 install -r ./requirements.txt

# Set environment variable based on the ARG
ENV mode=$ENV


# Verbose logging to ensure each step is tracked
ENTRYPOINT [ "sh", "./prod.sh" ]

# Entry point to run the Python application
# CMD [ "python3", "./src/" ]
