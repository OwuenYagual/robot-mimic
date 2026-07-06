#Cámara
CAMERA_INDEX   = 0        # Índice de la webcam (0 = cámara por defecto)
FRAME_WIDTH    = 640      # Ancho de salida del frame (píxeles)
FRAME_HEIGHT   = 480      # Alto de salida del frame (píxeles)

#Preprocesamiento (M1)
APPLY_GAUSSIAN  = False           # True para activar filtro gaussiano
GAUSSIAN_KERNEL = (5, 5)          # Tamaño del kernel (debe ser impar)

#MediaPipe Pose (M2)
MIN_DETECTION_CONFIDENCE = 0.7    # Confianza mínima de detección
MIN_TRACKING_CONFIDENCE  = 0.7    # Confianza mínima de tracking
LANDMARKS_OF_INTEREST    = [11, 13, 15, 17, 19, 21]  # Brazo derecho

#Cálculo de ángulos (M3)
SMOOTHING_WINDOW = 5              # Ventana de media móvil (frames)
VISIBILITY_THRESHOLD = 0.5        # Visibilidad mínima de landmark válido
GRIPPER_D_MIN = 20                # Distancia mínima en px (pinza cerrada)
GRIPPER_D_MAX = 120               # Distancia máxima en px (pinza abierta)

#Brazo robótico virtual (M4)
LINK_SHOULDER   = 120             # Longitud eslabón hombro (px)
LINK_FOREARM    = 100             # Longitud eslabón antebrazo (px)
LINK_HAND       = 60              # Longitud eslabón mano (px)
LINK_GRIPPER    = 30              # Longitud rama del gripper (px)
ROBOT_BASE_X    = 80              # Posición X de la base del brazo virtual
ROBOT_BASE_Y    = 420             # Posición Y de la base del brazo virtual