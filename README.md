# **Reconocimiento de dígitos manuscritos mediante un modelo MLP y CNN**
Este proyecto implementa dos tipos de redes neuronales: Red neuronal multicapa (MLP) y red neuronal convolucional (CNN). Entrenados con el dataset MNIST, para el reconocimiento de dígitos manuscritos. Ambos modelos entrenados serán implementados en una página web para poder probarlos.
## **Estructura del proyecto**
```plaintext
PAM/
├── config/                         # Archivos de configuración
├── data/                           # Datos originales y procesados
│   ├── processed/                  # Datos preprocesados para MLP y CNN
│   ├── raw/                        # Datos originales descargados en formato RAW
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
## Tecnologías utilizadas
Para todo lo concerniente a las redes neuronales, el manejo de MNIST y pruebas, se ha utilizado Python y los siguientes módulos de este:
- PyYAML
- TensorFlow
- Numpy
- Matplotlib
## Características de las redes neuronales
### MLP

### CNN
## Resultados