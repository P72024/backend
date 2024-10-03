#!/bin/sh

if [ "$ENV" = "production" ]; then
  echo "Running in production mode..."
  pip3 install nvidia-cudnn-cu12==8.9.5.30
else
  echo "Running in development mode..."
  # Any development-related setup or commands
fi



cd /app
pip3 install -r ./requirements.txt
python3 -u ./src/
