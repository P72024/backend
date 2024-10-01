#!/bin/sh

if [ "$ENV" = "production" ]; then
  echo "Running in production mode..."
  ls /usr/local/cuda/flat; if [ $? -eq 2 ]; then \
      echo "/usr/local/cuda does not exist." \
      echo "Production mode: Installing nvidia-cudnn"; \
      wget https://raw.githubusercontent.com/NVIDIA/build-system-archive-import-examples/refs/heads/main/parse_redist.py -O /usr/local/cuda/parse_redist.py; \
      chmod +x /usr/local/cuda/parse_redist.py; \
      echo "Running parse_redist.py script..."; \
      cd /usr/local/cuda; \
      python3 -u ./parse_redist.py --url https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_8.9.5.json --os linux --arch aarch64; \
      echo "Finished running parse_redist.py"; \

    else \
      echo "/usr/local/cuda already exists. Skipping nvidia-cudnn installation."; \
    fi;
  # Any production-related setup or commands
else
  echo "Running in development mode..."
  # Any development-related setup or commands
fi

# Symlink all files and symlinks from the CUDA include folder to /usr/include
find /usr/local/cuda/flat/linux-aarch64/cuda12/include -exec ln -s {} /usr/include/ \;

# Symlink all files and symlinks from the CUDA lib folder to /usr/lib
find /usr/local/cuda/flat/linux-aarch64/cuda12/lib -exec ln -s {} /usr/lib/ \;

cd /app
pip3 install -r ./requirements.txt
python3 -u ./src/