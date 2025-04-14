document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const questionForm = document.getElementById('questionForm');
    const uploadStatus = document.getElementById('uploadStatus');
    const answerDiv = document.getElementById('answer');
    const askButton = document.getElementById('askButton');

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const formData = new FormData();
        const fileInput = document.getElementById('pdfFile');
        formData.append('file', fileInput.files[0]);

        uploadStatus.style.display = 'block';
        uploadStatus.innerHTML = '<div class="loading"></div> Uploading...';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (response.ok) {
                uploadStatus.innerHTML = '<div class="alert alert-success">' + data.message + '</div>';
            } else {
                uploadStatus.innerHTML = '<div class="alert alert-danger">' + data.error + '</div>';
            }
        } catch (error) {
            uploadStatus.innerHTML = '<div class="alert alert-danger">Error uploading file</div>';
        }
    });

    questionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const question = document.getElementById('question').value;
        askButton.disabled = true;
        answerDiv.style.display = 'block';
        answerDiv.innerHTML = '<div class="loading"></div> Processing...';

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question })
            });
            const data = await response.json();
            
            if (response.ok) {
                answerDiv.innerHTML = '<strong>Réponse:</strong><br>' + data.answer;
            } else {
                answerDiv.innerHTML = '<div class="alert alert-danger">' + data.error + '</div>';
            }
        } catch (error) {
            answerDiv.innerHTML = '<div class="alert alert-danger">Error processing question</div>';
        } finally {
            askButton.disabled = false;
        }
    });
}); 