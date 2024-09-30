FROM python:3

WORKDIR /app/

COPY . .

ENTRYPOINT [ "python3", "./src/" ]