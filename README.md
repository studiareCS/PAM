# **Reconocimiento de dígitos manuscritos mediante un modelo MLP y CNN**
Este proyecto implementa dos tipos de redes neuronales: Red neuronal multicapa (MLP) y red neuronal convolucional (CNN). Entrenados con el dataset MNIST, para el reconocimiento de dígitos manuscritos. Ambos modelos entrenados serán implementados en una página web para poder probarlos.
## **Estructura del proyecto**
```plaintext
PAM/
├── backend/                        # Archivos para el backend
├── ├── app.py                      # Maneja el funcionamiento del servidor Flask
├── config/                         # Archivos de configuración
├── ├── config.yaml                 # Configuracion de los modelos MLP y CNN
├── data/                           # Datos originales y procesados
│   ├── processed/                  # Datos preprocesados para MLP y CNN
│   ├── raw/                        # Datos originales descargados en formato RAW
├── frontend/                       # Archivos para el frontend de la página web
├── ├── static/                     # Archivos estáticos (CSS, JavaScript)
├── ├── ├── script.js               # Maneja la lógica interactiva de la página
├── ├── ├── styles.css              # Establece los estilos visuales de la página
├── ├── index.html                  # Estructura básica de la página web
├── notebooks/                      # Jupyter notebooks para experimentación
├── scripts/                        # Scripts para el manejo de datos, entrenamiento, etc.
│   ├── data_pipeline.py            # Descarga y preprocesamiento de datos
│   ├── train_models.py             # Entrenamiento de los modelos
├── src/                            # Código fuente para el proyecto
|   ├── __init__.py                 # Convierte a src/ en un paquete
│   ├── data_preprocessing.py       # Definición del modelo MLP
│   ├── models.py                   # Definición del modelo CNN
├── README.md                       # Documentación del proyecto
├── requirements.txt                # Dependencias del proyecto
├── .gitignore                      # Archivos o carpetas a ignorar
```
## Instalación
*Se requiere tener Python instalado antes*
1. Clonar el repositorio
```bash
git clone https://github.com/studiareCS/PAM.git
cd PAM
```
2. Crear un entorno virtual (opcional)
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
```
3. Instalar dependencias 
```bash
pip install -r requirements.txt
```
4. Descargar, preprocesar y guardar el dataset MNIST
```bash
python -m scripts.data_pipeline
```
5. Entrenamiento de redes neuronales (MLP y CNN)
```bash
python -m scripts.train_models
```
## Uso
Asegurarse de encontrarse dentro del directorio PAM/
1. Ejecutar el servidor Flask
```bash
python backend/app.py
```
2. Si se ha ejecutado de buena forma, debería encontrarse lo siguiente:
```
 * Running on http://127.0.0.1:5000
```
3. Entrar a http://127.0.0.1:5000 para poder utilizar la página web con los modelos ya implementados para las predicciones (MLP y CNN)
## Tecnologías utilizadas
Para todo lo concerniente a las redes neuronales, el manejo de MNIST y pruebas, se ha utilizado Python y los siguientes módulos de este:
- PyYAML
- TensorFlow
- Numpy
- Matplotlib
## Características de las redes neuronales
Para ambos modelos se utiliza la función de costo "categorical cross entropy" y el optimizador "Adam".
### MLP
Input layer: 
Flatten layer: 
84 unidades (una para el valor de cada pixel de la imagen 28x28)
Hidden layers:
- Fully coneccted layer: 256 neuronas (ReLU)
- Fully coneccted layer: 128 neuronas (ReLU)
- Fully coneccted layer: 64 neuronas (ReLU)
Output layer:
Fully connected layer: 10 neuronas (Softmax)
### CNN
Input layer:
Layer que recibe las imagenes de 28x28 (1 canal por estar en escala de grises)
Hidden layers:
- Convolutional layer: 32 kernels de 3x3 (ReLU)
- Max pooling layer: Kernel de 2x2
- Convolutional layer: 64 kernels de 3x3 (ReLU)
- Max pooling layer: Kernel de 2x2
- Flatten layer (Aplana los valores de los pixeles de la imagen)
- Fully connected layer: 128 neuronas (ReLU)
Output layer:
Fully connected layer: 10 neuronas (Softmax)
## Resultados
*Obtenidos con la configuración presentada anteriormente de cada modelo*
Precisión del modelo MLP: 99.21%
Precisión del modelo CNN: 99.85%