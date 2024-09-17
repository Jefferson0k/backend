# Proyecto de Tesis: Plataforma de Traducción y Síntesis de Voz

Este proyecto de tesis proporciona una plataforma en línea para la traducción y la síntesis de voz. El sistema está compuesto por un backend en Django y un frontend desarrollado con Next.js y TypeScript.

## Estructura del Proyecto

- **Backend:** [GitHub - backend](https://github.com/Jefferson0k/backend.git)
- **Frontend:** [GitHub - zoom-clone](https://github.com/Jefferson0k/zoom-clone.git)

## Backend

El backend de la aplicación está desarrollado con **Django** y se encarga de la lógica de traducción y la generación de voz.

### Características

- Traducción de texto a múltiples idiomas.
- Síntesis de voz a partir del texto traducido.
- API para la comunicación con el frontend.

### Requisitos

- Python 3.x
- Django
- TensorFlow
- OpenCV
- Mediapipe
- Pygame

### Instalación

1. Clona el repositorio:
    ```bash
    git clone https://github.com/Jefferson0k/backend.git
    ```

2. Crea y activa un entorno virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate   # En Linux/Mac
    .\venv\Scripts\activate    # En Windows
    ```

3. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

4. Realiza las migraciones:
    ```bash
    python manage.py migrate
    ```

5. Inicia el servidor:
    ```bash
    python manage.py runserver
    ```

## Frontend

El frontend de la aplicación está desarrollado con Next.js y TypeScript, y proporciona la interfaz de usuario para interactuar con la plataforma de traducción y síntesis de voz.

### Características

- Interfaz de usuario para ingresar el texto a traducir.
- Visualización del texto traducido.
- Reproducción de la voz sintetizada.

### Requisitos

- Node.js
- npm (o yarn)

### Instalación

1. Clona el repositorio:
    ```bash
    git clone https://github.com/Jefferson0k/zoom-clone.git
    ```

2. Navega al directorio del proyecto:
    ```bash
    cd zoom-clone
    ```

3. Instala las dependencias:
    ```bash
    npm install
    # o usando yarn
    yarn install
    ```

4. Inicia la aplicación:
    ```bash
    npm run dev
    # o usando yarn
    yarn dev
    ```

## Contacto

Para cualquier consulta sobre el código o el proyecto, por favor contacta a Jefferson Covenas en [jefersoncovenas7@gmail.com](mailto:jefersoncovenas7@gmail.com).
