"""
m4_robot.py
Módulo 4 – Simulación del Brazo Robótico Virtual (demo)
Proyecto: Sistema de Imitación de Movimientos del Brazo Humano
Curso:    Procesamiento Digital de Imágenes – DIP 2026 PAO-I
ESPOL – FIEC

Renderiza un brazo robótico de 3 eslabones + gripper usando
únicamente primitivas OpenCV (líneas y círculos).
Los ángulos vienen directamente de M3.
"""

import cv2
import math
import numpy as np

from config.settings import (
    LINK_SHOULDER,
    LINK_FOREARM,
    LINK_HAND,
    LINK_GRIPPER,
    ROBOT_BASE_X,
    ROBOT_BASE_Y,
)


# ── Colores BGR ───────────────────────────────────────────────────────────
COLOR_BASE      = (180, 180, 180)   # gris
COLOR_LINK      = (255, 140,   0)   # azul petróleo
COLOR_JOINT     = (255, 255, 255)   # blanco
COLOR_EFFECTOR  = (0,   200, 255)   # amarillo
COLOR_GRIPPER   = (0,   165, 255)   # naranja
COLOR_LABEL     = (200, 200, 200)   # gris claro


class RobotArm:
    """
    Módulo M4: dibuja el brazo robótico virtual sobre el frame BGR
    usando cinemática directa 2D.

    Cadena cinemática:
        BASE → eslabón 1 (hombro) → J1 → eslabón 2 (antebrazo)
             → J2 → eslabón 3 (mano) → J3 → gripper (pinza)

    Args:
        base_x : coordenada X de la base en píxeles.
        base_y : coordenada Y de la base en píxeles.
    """

    def __init__(self, base_x: int = ROBOT_BASE_X, base_y: int = ROBOT_BASE_Y):
        self.base_x = base_x
        self.base_y = base_y

    # ------------------------------------------------------------------ #
    # Interfaz pública                                                     #
    # ------------------------------------------------------------------ #

    def draw(self, frame: np.ndarray, angles: dict | None) -> np.ndarray:
        """
        Renderiza el brazo robótico sobre el frame con los ángulos dados.

        Args:
            frame  : imagen BGR donde se dibuja el brazo.
            angles : dict con "shoulder", "elbow", "gripper" (de M3).
                     Si es None dibuja el brazo en posición de reposo.

        Returns:
            Frame BGR con el brazo superpuesto.
        """
        if angles is None:
            shoulder_deg = 90.0
            elbow_deg    = 160.0
            gripper_norm = 0.0
        else:
            shoulder_deg = angles.get("shoulder", 90.0)
            elbow_deg    = angles.get("elbow",    160.0)
            gripper_norm = angles.get("gripper",  0.0)

        # ── Cinemática directa 2D ─────────────────────────────────────
        # Convertir ángulos a radianes
        # El eje Y de OpenCV apunta hacia abajo, por eso restamos en Y
        theta1 = math.radians(shoulder_deg)   # hombro respecto a vertical
        theta2 = math.radians(elbow_deg)       # codo respecto al eslabón 1

        # Base
        p0 = (self.base_x, self.base_y)

        # J1 (codo)
        p1 = (
            int(p0[0] + LINK_SHOULDER * math.sin(theta1)),
            int(p0[1] - LINK_SHOULDER * math.cos(theta1)),
        )

        # J2 (muñeca)
        angle_accum = theta1 + (theta2 - math.pi)   # ángulo acumulado
        p2 = (
            int(p1[0] + LINK_FOREARM * math.sin(angle_accum)),
            int(p1[1] - LINK_FOREARM * math.cos(angle_accum)),
        )

        # J3 (efector)
        p3 = (
            int(p2[0] + LINK_HAND * math.sin(angle_accum)),
            int(p2[1] - LINK_HAND * math.cos(angle_accum)),
        )

        # Apertura del gripper en píxeles
        gripper_open_px = int(LINK_GRIPPER * gripper_norm) + 6

        # Dirección perpendicular al último eslabón (para abrir la pinza)
        dx = p3[0] - p2[0]
        dy = p3[1] - p2[1]
        norm = math.sqrt(dx**2 + dy**2) + 1e-9
        perp_x = int(-dy / norm * gripper_open_px)
        perp_y = int( dx / norm * gripper_open_px)

        # Puntas del gripper
        g_top = (p3[0] + perp_x, p3[1] + perp_y)
        g_bot = (p3[0] - perp_x, p3[1] - perp_y)

        # ── Dibujo ────────────────────────────────────────────────────
        out = frame.copy()

        # Base (rectángulo)
        cv2.rectangle(
            out,
            (p0[0] - 18, p0[1]),
            (p0[0] + 18, p0[1] + 12),
            COLOR_BASE, -1,
        )
        cv2.rectangle(
            out,
            (p0[0] - 18, p0[1]),
            (p0[0] + 18, p0[1] + 12),
            COLOR_JOINT, 1,
        )

        # Eslabón 1: base → J1
        cv2.line(out, p0, p1, COLOR_LINK, 4, cv2.LINE_AA)

        # Eslabón 2: J1 → J2
        cv2.line(out, p1, p2, COLOR_LINK, 3, cv2.LINE_AA)

        # Eslabón 3: J2 → J3
        cv2.line(out, p2, p3, COLOR_LINK, 2, cv2.LINE_AA)

        # Gripper: dos ramas desde J3
        cv2.line(out, p3, g_top, COLOR_GRIPPER, 2, cv2.LINE_AA)
        cv2.line(out, p3, g_bot, COLOR_GRIPPER, 2, cv2.LINE_AA)

        # Articulaciones (círculos)
        for pt, r in [(p0, 7), (p1, 6), (p2, 5), (p3, 4)]:
            cv2.circle(out, pt, r, COLOR_JOINT,  -1)
            cv2.circle(out, pt, r, COLOR_LINK,    1)

        # Etiquetas
        labels = {p0: "BASE", p1: "J1", p2: "J2", p3: "EF"}
        for pt, label in labels.items():
            cv2.putText(
                out, label,
                (pt[0] + 8, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                COLOR_LABEL, 1, cv2.LINE_AA,
            )

        # Ángulos sobre las articulaciones
        cv2.putText(
            out, f"{shoulder_deg:.0f}",
            (p1[0] - 28, p1[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            COLOR_EFFECTOR, 1, cv2.LINE_AA,
        )
        cv2.putText(
            out, f"{elbow_deg:.0f}",
            (p2[0] - 28, p2[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            COLOR_EFFECTOR, 1, cv2.LINE_AA,
        )
        cv2.putText(
            out, f"{gripper_norm*100:.0f}%",
            (p3[0] + 8, p3[1] + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            COLOR_GRIPPER, 1, cv2.LINE_AA,
        )

        return out


# ---------------------------------------------------------------------- #
# Demo autónomo: mueve el brazo con sliders (sin cámara)                 #
# Ejecutar: python -m modules.m4_robot                                   #
# ---------------------------------------------------------------------- #

if __name__ == "__main__":

    DEMO_W, DEMO_H = 640, 480

    cv2.namedWindow("M4 - Robot Demo")
    cv2.createTrackbar("Hombro",  "M4 - Robot Demo", 90,  180, lambda x: None)
    cv2.createTrackbar("Codo",    "M4 - Robot Demo", 160, 180, lambda x: None)
    cv2.createTrackbar("Gripper", "M4 - Robot Demo", 0,   100, lambda x: None)

    robot = RobotArm(base_x=120, base_y=400)

    print("Demo M4 – usa los sliders para mover el brazo.")
    print("Presiona 'q' para salir.")

    while True:
        canvas = np.zeros((DEMO_H, DEMO_W, 3), dtype=np.uint8)

        # Leer sliders
        shoulder_deg = cv2.getTrackbarPos("Hombro",  "M4 - Robot Demo")
        elbow_deg    = cv2.getTrackbarPos("Codo",    "M4 - Robot Demo")
        gripper_pct  = cv2.getTrackbarPos("Gripper", "M4 - Robot Demo")

        angles = {
            "shoulder": float(shoulder_deg),
            "elbow":    float(elbow_deg),
            "gripper":  gripper_pct / 100.0,
        }

        frame = robot.draw(canvas, angles)

        # Instrucciones
        cv2.putText(frame, "M4 - Brazo Robotico Virtual (demo)",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Usa los sliders para mover el brazo",
            (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1, cv2.LINE_AA)

        cv2.imshow("M4 - Robot Demo", frame)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            print("Saliendo...")
            break

    cv2.destroyAllWindows()