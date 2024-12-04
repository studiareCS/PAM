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

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageData.split(',')[1] })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('predictionResult').textContent = `Predicción: ${data.prediction}`;
    })
    .catch(error => console.error('Error al realizar la predicción:', error));
});