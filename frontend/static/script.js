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