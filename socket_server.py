from flask import Flask, render_template, url_for, send_from_directory
from flask_socketio import SocketIO, send, emit
from time import sleep
import python_speech_features
import numpy as np
count=0
app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app,async_handlers=True)

@app.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('static', path)


@app.route("/")

def hello():
    return render_template('index.html')
@socketio.on('message')
def handle_message(message):
    # global count
    # count+=1
    # emit('stream',{'audio': 111})
    print(type(message))
    print('received message: ',np.frombuffer(message['data'],np.int8).shape)
    


if __name__ == '__main__':
    # app.run(ssl_context='adhoc')
    # print('asda')
    socketio.run(app,host='0.0.0.0', port=5000)
