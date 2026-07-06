import cv2
import numpy as np
from config.settings import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    APPLY_GAUSSIAN,
    GAUSSIAN_KERNEL,
)


class CaptureModule:
    """
    Módulo M1: adquiere frames de la cámara y los preprocesa
    para su uso en el módulo M2 (MediaPipe Pose).

    Pipeline interno:
        VideoCapture → resize → BGR→RGB → [gaussiano opcional]

    Attributes:
        cap       : objeto cv2.VideoCapture activo
        width     : ancho de salida en píxeles
        height    : alto de salida en píxeles
        apply_blur: activar/desactivar filtro gaussiano
    """

    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.width = FRAME_WIDTH
        self.height = FRAME_HEIGHT
        self.apply_blur = APPLY_GAUSSIAN

        if not self.cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara (índice {CAMERA_INDEX}). "
                "Verifica que esté conectada y no esté en uso."
            )

        # Solicitar resolución nativa al driver (puede ser ignorada por el SO)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    # ------------------------------------------------------------------ #
    # Interfaz pública                                                     #
    # ------------------------------------------------------------------ #

    def read(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """
        Captura un frame y aplica el pipeline de preprocesamiento.

        Returns:
            ok      : True si el frame fue capturado correctamente.
            bgr     : frame en BGR (para renderizado con OpenCV).
            rgb     : frame en RGB preprocesado (entrada para MediaPipe).
                      None si ok es False.
        """
        ok, bgr = self.cap.read()
        if not ok:
            return False, None, None

        bgr = self._resize(bgr)
        bgr = cv2.flip(bgr, 1)  # espejo horizontal para simular cámara frontal

        if self.apply_blur:
            bgr = self._gaussian_blur(bgr)

        rgb = self._bgr_to_rgb(bgr)
        return True, bgr, rgb

    def release(self) -> None:
        """Libera la cámara y cierra los recursos de VideoCapture."""
        if self.cap.isOpened():
            self.cap.release()

    def is_opened(self) -> bool:
        """Retorna True si la cámara está activa."""
        return self.cap.isOpened()

    # ------------------------------------------------------------------ #
    # Pasos del pipeline (privados)                                        #
    # ------------------------------------------------------------------ #

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """
        Redimensiona el frame a (FRAME_WIDTH × FRAME_HEIGHT) usando
        interpolación bicúbica para preservar la calidad de bordes.

        Args:
            frame: imagen BGR de entrada (cualquier resolución).

        Returns:
            Imagen BGR redimensionada a (width, height).
        """
        return cv2.resize(
            frame,
            (self.width, self.height),
            interpolation=cv2.INTER_CUBIC,
        )

    def _gaussian_blur(self, frame: np.ndarray) -> np.ndarray:
        """
        Aplica un filtro gaussiano para atenuar ruido de alta frecuencia
        antes de la inferencia de MediaPipe.

        Args:
            frame: imagen BGR.

        Returns:
            Imagen BGR suavizada.
        """
        return cv2.GaussianBlur(frame, GAUSSIAN_KERNEL, sigmaX=0)

    @staticmethod
    def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """
        Convierte el frame de BGR (formato OpenCV) a RGB
        (formato requerido por MediaPipe Pose).

        Args:
            frame: imagen BGR uint8.

        Returns:
            Imagen RGB uint8.
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------ #
    # Context manager (uso con 'with')                                     #
    # ------------------------------------------------------------------ #

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False