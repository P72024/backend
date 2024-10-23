import socketio

sio = socketio.AsyncServer()

@sio.on(event='hello')
def hello():
    print('hello world')
