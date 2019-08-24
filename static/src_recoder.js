var p = document.getElementById('p');
var socket = io('http://127.0.0.1:8080/');
let textt = document.getElementById('tt');
var reader = new FileReader();
var recoder
var blobEvent
window.AudioContext = window.AudioContext || window.webkitAudioContext;
navigator.getUserMedia =  navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia

// navigator.getUserMedia || navigator.webkitGetUserMedia;
             
var s = document.getElementById('s');
var timer;
// ,sampleRate:16000
navigator.getUserMedia({audio: true, SampleRate:16000}, function(stream) {
context=new AudioContext()
source = context.createBufferSource();
recoder = new MediaRecorder(stream)
console.log(recoder)
recoder.start(3000)

// blobEvent = new BlobEvent(recoder);
// console.log(microphone.mediaStream)

socket.on('stream', function(s){
    console.log(s)
});

reader.onload = function(event){
    console.log(reader.result);//内容就在这里
    timer=reader.result
    context.decodeAudioData(reader.result,function(decodedData) {
        source.buffer = decodedData;
        source.connect(context.destination);
        source.loop = true;
    console.log(decodedData.getChannelData(0))
 // use the decoded data here
    socket.emit('message', {'data':decodedData.getChannelData(0)});
    },
function(e){ console.log("Error with decoding audio data" + e.err); });
    
  };
recoder.ondataavailable = function(e) {
    console.log(e.data)
    document.getElementById('player').src = URL.createObjectURL(e.data);
    reader.readAsArrayBuffer(e.data)
  }
p.onclick = function(){
// recoder.stop()
// recoder.requestData()
// recoder.start()
};

}, function(){
  tt.innerHTML='error';
});

