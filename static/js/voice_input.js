// Voice input functionality using Web Speech API
document.addEventListener('DOMContentLoaded', function () {
    const voiceBtn = document.getElementById('voiceBtn');
    const queryInput = document.getElementById('queryInput');

    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        voiceBtn.disabled = true;
        voiceBtn.title = "Speech recognition not supported in this browser";
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    recognition.continuous = false;
    recognition.interimResults = false;

    // Set dynamic language based on selection
    const langSelect = document.getElementById('languageSelect');
    const updateLang = () => {
        if (langSelect) {
            const currentLang = langSelect.value;
            recognition.lang = currentLang === "hi" ? "hi-IN" : currentLang === "te" ? "te-IN" : "en-IN";
            console.log("Speech recognition language set to:", recognition.lang);
        }
    };

    if (langSelect) {
        langSelect.addEventListener('change', updateLang);
        updateLang();
    } else {
        recognition.lang = 'en-IN';
    }

    voiceBtn.addEventListener('click', function () {
        if (voiceBtn.classList.contains('listening')) {
            recognition.stop();
            voiceBtn.classList.remove('listening', 'btn-danger');
            voiceBtn.classList.add('btn-outline-secondary');
            voiceBtn.innerHTML = '<i class="bi bi-mic"></i> Voice';
        } else {
            recognition.start();
            voiceBtn.classList.add('listening', 'btn-danger');
            voiceBtn.classList.remove('btn-outline-secondary');
            voiceBtn.innerHTML = '<i class="bi bi-mic-fill"></i> Listening...';
        }
    });

    recognition.onresult = function (event) {
        const transcript = event.results[0][0].transcript;
        queryInput.value = transcript;
        recognition.stop();
        voiceBtn.classList.remove('listening', 'btn-danger');
        voiceBtn.classList.add('btn-outline-secondary');
        voiceBtn.innerHTML = '<i class="bi bi-mic"></i> Voice';
    };

    recognition.onerror = function (event) {
        console.error('Speech recognition error', event.error);
        voiceBtn.classList.remove('listening', 'btn-danger');
        voiceBtn.classList.add('btn-outline-secondary');
        voiceBtn.innerHTML = '<i class="bi bi-mic"></i> Voice';
    };

    recognition.onend = function () {
        if (voiceBtn.classList.contains('listening')) {
            voiceBtn.classList.remove('listening', 'btn-danger');
            voiceBtn.classList.add('btn-outline-secondary');
            voiceBtn.innerHTML = '<i class="bi bi-mic"></i> Voice';
        }
    };
});