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
## Instalación
## Uso
## Tecnologías utilizadas
## Características de las redes neuronales
## Resultados