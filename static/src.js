var p = document.getElementById('p');
var socket = io('http://127.0.0.1:5000/');
let textt = document.getElementById('tt');
var reader = new FileReader();
var recoder
var blobEvent
window.AudioContext = window.AudioContext || window.webkitAudioContext;
navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia;
             
var s = document.getElementById('s');
var timer;

navigator.getUserMedia({audio: true}, function(stream) {

recoder = new MediaRecorder(stream)
recoder.start()

// blobEvent = new BlobEvent(recoder);
// console.log(microphone.mediaStream)

socket.on('stream', function(s){
    console.log(s)
});

reader.onload = function(event){
    console.log(reader.result);//内容就在这里
    timer=reader.result
    socket.emit('message', {'data':reader.result});
  };
recoder.ondataavailable = function(e) {
    console.log(e.data)
    reader.readAsArrayBuffer(e.data)
  }
p.onclick = function(){

recoder.requestData()
};
// var analyser = context.createAnalyser();
// microphone.connect(analyser);
// analyser.connect(context.destination);

// analyser.fftSize = 32;
// var bufferLength = analyser.frequencyBinCount;
// var dataArray = new Uint8Array(analyser.fftSize);
// //analyser.getByteFrequencyData(dataArray);

// s.onclick = function(){
//   clearTimeout(timer);
// };


// update();
// tt.innerHTML='vvvv'
// function update(){
//   console.log(dataArray);
//   tt.innerHTML=dataArray
//   analyser.getByteFrequencyData(dataArray);
//   timer = setTimeout(update,20);
// }

}, function(){
  tt.innerHTML='error';
});

