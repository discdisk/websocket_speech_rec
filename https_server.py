from flask import Flask, render_template, url_for, send_from_directory
from flask_socketio import SocketIO, send, emit
from time import sleep
import python_speech_features
count=0
app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app,async_handlers=True)

@app.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('static', path)


@app.route("/")
def hello():
    return render_template('index.html')

    


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=8080, ssl_context=('cert.pem', 'key.pem'))
    # print('asda')
    # socketio.run(app,debug=True,host='0.0.0.0',ssl_context='adhoc')
