#!/bin/sh

if [ "$ENV" = "production" ]; then
  echo "Running in production mode..."
  pip3 install nvidia-cudnn-cu12==8.9.5.30
  export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
else
  echo "Running in development mode..."
  # Any development-related setup or commands
fi



cd /app
pip3 install -r ./requirements.txt
python3 -u ./src/
