var p = document.getElementById('p');
var socket = io('http://127.0.0.1:8080/');
let textt = document.getElementById('tt');
var reader = new FileReader();
window.AudioContext = window.AudioContext || window.webkitAudioContext;
MediaDevices.getUserMedia =  MediaDevices.getUserMedia || MediaDevices.webkitGetUserMedia || MediaDevices.mozGetUserMedia || navigator.msGetUserMedia


             
var s = document.getElementById('s');
var chunks = [];
navigator.getUserMedia({audio: true, sampleRate:16000}, function(stream) {
  context=new AudioContext({sampleRate: 16000})
  mic=context.createMediaStreamSource(stream)
  
  scriptNode = context.createScriptProcessor(16384, 1, 1);

  mic.connect(scriptNode)
  scriptNode.connect(context.destination)
  console.log(context.sampleRate)

    

socket.on('result', function(s){
    console.log(s)
    tt.innerHTML=s['data']
    });

scriptNode.onaudioprocess = function(audioProcessingEvent) {
  // The input buffer is the song we loaded earlier
  var inputBuffer = audioProcessingEvent.inputBuffer;
    console.log(inputBuffer.getChannelData(0))
  // The output buffer contains the samples that will be modified and played
  var outputBuffer = audioProcessingEvent.outputBuffer;

  // Loop through the output channels (in this case there is only one)
  for (var channel = 0; channel < outputBuffer.numberOfChannels; channel++) {
    var inputData = inputBuffer.getChannelData(channel);
    var outputData = outputBuffer.getChannelData(channel);
    outputData = inputData
    socket.emit('message', {'data':inputData.buffer});

    // Loop through the 4096 samples
    // for (var sample = 0; sample < inputBuffer.length; sample++) {
    //   // make output equal to the same as the input
    //   outputData[sample] = inputData[sample];
   
    // }
      
  }
}


p.onclick = function(){
  tt.innerHTML='errorasds';
};

}, function(){
  tt.innerHTML='error';
});

