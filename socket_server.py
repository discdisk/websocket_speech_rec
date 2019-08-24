from flask import Flask, render_template, url_for, send_from_directory
from flask_socketio import SocketIO, send, emit
from time import sleep
from sound_process import preprocess
from neural_network.models.model_maxpooling import RNN
import chainer.functions as F
import chainer
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
import io
import pickle

def merge_drop(arr:list):
    new=[]
    prev=None
    i=0
    for now in arr:
        if now!=prev and now!='blank':
            new.append(now)
            prev=now
    print(new)
    return ''.join(new)



# decode_dic=pickle.load(open('SPS_npData_word_dic.pkl','rb'))
# decode_dic={y:x for x,y in decode_dic.items()}
# model = RNN(n_lstm_layers=5, n_mid_units=320, n_out=len(decode_dic), win_size=7, batch_size=1, att_units_size=int(320/4), dropout=0)
# model = chainer.links.Classifier(model)
# chainer.serializers.load_npz('model_SPS',model)
# model=model.predictor

char_dic=pickle.load(open('APS_npData_char_dic.pkl','rb'))
decode_dic={y:x for x,y in char_dic.items()}
model = RNN(n_lstm_layers=5, n_mid_units=320, n_out=len(decode_dic), win_size=7, batch_size=1, att_units_size=int(320/4), dropout=0)
chainer.serializers.load_npz('model_char_300000',model)

model.stream_init()
reserve=[]
out_put=[]
wavfile=np.array([],dtype=np.float32)

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
    global reserve
    global wavfile
    global out_put
    # print(reserve)
    data=np.frombuffer(message['data'],np.float32)
    # print(data.shape)

    if reserve == []:
        data=np.append(reserve,data)
    cutoff=(len(data)-(400-160))%160+(400-160)
    reserve=data[-cutoff:]
    data=data[:-cutoff]

    wavfile=np.append(wavfile,data)
    sf.write('1.wav',wavfile,16000)

    feature = preprocess((data*(2**15)))
    # print('received message: ', feature.shape)
    # print(feature)

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            # print(id(model))
            result=model.test_stream(feature)
    # print(result)
    if type(result) is not type(None):
        out_put+=[decode_dic[int(F.argmax(r).data) if F.max(F.softmax(r)).data>0.5 else 0] for r in result ]
        emit('result',{'data':merge_drop(out_put)})
        
    


if __name__ == '__main__':
    # app.run(ssl_context='adhoc')
    # print('asda')
    socketio.run(app,debug=True,host='0.0.0.0', port=8080)
