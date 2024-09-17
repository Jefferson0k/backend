from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tempfile
import os

# Verificar si estamos en un entorno que soporta audio
audio_enabled = not os.environ.get('RENDER') and not os.environ.get('DISABLE_AUDIO')

# Si el entorno soporta audio, inicializamos Pygame
if audio_enabled:
    import pygame
    pygame.mixer.init()
else:
    print("Entorno sin acceso a audio, Pygame no se inicializa.")

# Cargar el modelo entrenado
model = load_model('lsp/media/resultados/action_recognition_model.h5')

# Inicializar Mediapipe
mp_holistic = mp.solutions.holistic

# Diccionario que asocia cada acción con su archivo de audio
audio_files = {
    'hola': 'lsp/media/audios/hola.mp3',
    'gracias': 'lsp/media/audios/gracias.mp3',
    'adios': 'lsp/media/audios/adios.mp3'
}

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

@csrf_exempt
def recognize_actions_from_video(request):
    if request.method == 'POST' and request.FILES.get('file'):
        video_file = request.FILES['file']
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        for chunk in video_file.chunks():
            temp_file.write(chunk)
        temp_file.close()
        
        cap = cv2.VideoCapture(temp_file.name)
        frames = []
        sequence_length = 30  # Número de frames que el modelo espera
        
        action_played = None  # Variable para rastrear la acción que se ha reproducido
        response = {'action': 'señal no detectada'}  # Valor predeterminado de la respuesta
        
        with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
            action_detected = False  # Bandera para saber si ya se detectó una acción
            while cap.isOpened() and not action_detected:
                ret, frame = cap.read()
                if not ret:
                    break
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                keypoints = extract_keypoints(results)
                frames.append(keypoints)
                
                # Mantener solo la cantidad de frames necesarios
                if len(frames) > sequence_length:
                    frames.pop(0)
                
                if len(frames) == sequence_length:
                    # Convertir la lista de frames en un numpy array
                    sequence = np.array(frames)
                    # Añadir una dimensión extra para el batch size
                    sequence = np.expand_dims(sequence, axis=0)
                    # Realizar la predicción
                    prediction = model.predict(sequence)
                    prediction_prob = np.max(prediction)
                    
                    # Lista de acciones
                    actions = ['hola', 'gracias', 'adios']
                    
                    if prediction_prob > 0.7:  # Umbral de confianza
                        action = actions[np.argmax(prediction)]
                        
                        if action != action_played:  # Solo reproducir si la acción es diferente
                            audio_file = audio_files.get(action, None)
                            if audio_file and os.path.exists(audio_file):
                                if audio_enabled:
                                    print(f"Reproduciendo audio: {audio_file}")
                                    pygame.mixer.music.load(audio_file)
                                    pygame.mixer.music.play()
                                action_played = action  # Actualizar la acción reproducida
                                response = {'action': action}
                                action_detected = True  # Indicar que se detectó una acción
                                
        cap.release()
        os.remove(temp_file.name)
        return JsonResponse(response)
    
    return JsonResponse({'error': 'No se ha enviado un archivo de video o el método de solicitud es incorrecto.'}, status=400)
