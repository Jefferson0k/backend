from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tempfile
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar el modelo entrenado
model = load_model('lsp/media/resultados/simulacion.h5')

# Inicializar Mediapipe
mp_holistic = mp.solutions.holistic

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
        logger.info(f"Recibido archivo de video de tamaño: {video_file.size} bytes")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        for chunk in video_file.chunks():
            temp_file.write(chunk)
        temp_file.close()
        
        cap = cv2.VideoCapture(temp_file.name)
        frames = []
        sequence_length = 30  # Número de frames que el modelo espera
        
        response = {'action': 'señal no detectada'}  # Valor predeterminado de la respuesta
        frame_count = 0
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                keypoints = extract_keypoints(results)
                frames.append(keypoints)
                
                # Mantener solo la cantidad de frames necesarios
                if len(frames) > sequence_length:
                    frames.pop(0)
                
                if len(frames) == sequence_length:
                    sequence = np.array(frames)
                    sequence = np.expand_dims(sequence, axis=0)
                    prediction = model.predict(sequence)
                    prediction_prob = np.max(prediction)
                    
                    actions = ['adios','bien','camiseta','diez','doce','gato','gracias','Hola','pantalones','rosado','trabajar','verde','zapatos']
                    
                    logger.info(f"Predicción raw: {prediction}")
                    logger.info(f"Probabilidad máxima: {prediction_prob}")
                    
                    if prediction_prob > 0.2:  # Umbral de confianza reducido
                        action = actions[np.argmax(prediction)]
                        response = {'action': action}
                        logger.info(f"Acción detectada: {action}")
                        break
        
        cap.release()
        os.remove(temp_file.name)
        
        logger.info(f"Total de frames procesados: {frame_count}")
        logger.info(f"Respuesta final: {response}")
        
        return JsonResponse(response)
    
    return JsonResponse({'error': 'No se ha enviado un archivo de video o el método de solicitud es incorrecto.'}, status=400)