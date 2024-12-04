// Obtener el canvas y el contexto
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Configuración del lienzo
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "black";
ctx.lineWidth = 15;
ctx.lineCap = "round";

// Dibujar en el lienzo
let isDrawing = false;

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
});

// Limpiar el lienzo
document.getElementById('clearBtn').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillRect(0, 0, canvas.width, canvas.height);  // Rellenar de blanco
    document.getElementById('predictionResult').textContent = '';
});

// Enviar la imagen al backend para la predicción
document.getElementById('predictBtn').addEventListener('click', () => {
    const imageData = canvas.toDataURL('image/png');
    
    // Convertir la cadena base64 a un objeto Blob (archivo)
    const byteCharacters = atob(imageData.split(',')[1]);  // Decodificar la base64 a bytes
    const byteArray = new Uint8Array(byteCharacters.length);

    // Convertir los bytes en un array de bytes
    for (let i = 0; i < byteCharacters.length; i++) {
        byteArray[i] = byteCharacters.charCodeAt(i);
    }

    // Crear un Blob con la imagen (en formato PNG)
    const blob = new Blob([byteArray], { type: 'image/png' });

    // Crear un objeto FormData para enviar el archivo
    const formData = new FormData();
    formData.append('file', blob, 'image.png');  // 'file' es el nombre del campo que espera el backend

    // Enviar la imagen al backend para la predicción
    fetch('/predict', {
        method: 'POST',
        body: formData  // Usar FormData para enviar el archivo
    })
    .then(response => response.json())
    .then(data => {
        // Verifica si ambos valores de predicción están presentes
        if (data.cnn_prediction !== undefined && data.mlp_prediction !== undefined) {
            document.getElementById('predictionResult').textContent = 
                `Predicción CNN: ${data.cnn_prediction}, Predicción MLP: ${data.mlp_prediction}`;
        } else {
            document.getElementById('predictionResult').textContent = 'Error en la predicción.';
        }
    })
    .catch(error => console.error('Error al realizar la predicción:', error));
});