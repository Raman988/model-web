function analyzeAudio() {
    const fileInput = document.getElementById('audioInput');
    const resultDiv = document.getElementById('result');
  
    const file = fileInput.files[0];
    if (!file) {
      alert('Please select an MP3 file.');
      return;
    }
  
    const reader = new FileReader();
    reader.onload = function(event) {
      const audioContext = new AudioContext();
      audioContext.decodeAudioData(event.target.result, function(buffer) {
        const audioData = buffer.getChannelData(0); // Mono audio
        const isNormal = checkAudio(audioData); // Function to check audio (you need to implement this)
        resultDiv.textContent = isNormal ? 'Sound is normal' : 'Sound is abnormal';
      });
    };
  
    reader.readAsArrayBuffer(file);
  }
  
  function checkAudio(audioData) {
    // Here you would send the audio data to your backend service with the trained model,
    // and receive the result (normal or abnormal) back.
    // For simplicity, you can just return a random result for now.
    return Math.random() < 0.5; // 50% chance of normal or abnormal
  }
  