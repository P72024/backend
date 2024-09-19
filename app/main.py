
from flask import Flask, jsonify
import pika

app = Flask(__name__)

# RabbitMQ setup
def send_to_queue(message):
    connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))  # RabbitMQ container hostname
    channel = connection.channel()
    channel.queue_declare(queue='task_queue', durable=True)  # Ensure queue is created
    channel.basic_publish(
        exchange='',
        routing_key='task_queue',
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=2,  # Make message persistent
        ))
    connection.close()

@app.route('/')
def home():
    return jsonify({"message": "Hello from the backend!"})

@app.route('/send')
def send_message():
    message = "Message to RabbitMQ!"
    send_to_queue(message)
    return jsonify({"status": "Message sent to RabbitMQ!"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
