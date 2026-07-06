"""
main.py
Pipeline completo: M1 + M2 + M3 + M4 (ventana separada)
Proyecto: Sistema de Imitación de Movimientos del Brazo Humano
Curso:    Procesamiento Digital de Imágenes – DIP 2026 PAO-I
ESPOL – FIEC
"""

import cv2
import time
import math
import numpy as np

from modules.m1_capture import CaptureModule
from modules.m2_mediapipe import PoseDetector
from modules.m3_angles  import AngleCalculator


# ── Parámetros del brazo virtual ─────────────────────────────────────────
ROBOT_W     = 400
ROBOT_H     = 480
BASE_X      = 200
BASE_Y      = 430
L1          = 120   # eslabón hombro
L2          = 100   # eslabón antebrazo
L3          = 70    # eslabón mano
L_GRIPPER   = 35    # longitud de cada rama de la pinza
L_FINGER    = 14    # longitud de la punta (dedo) del gripper


def draw_robot_window(angles: dict | None) -> np.ndarray:
    """
    Dibuja el brazo robótico completo sobre un canvas negro.
    Incluye base, 3 eslabones, gripper con dos ramas y puntas.
    """
    canvas = np.zeros((ROBOT_H, ROBOT_W, 3), dtype=np.uint8)

    if angles is None:
        sh = 90.0
        el = 160.0
        gr = 0.0
    else:
        sh = angles.get("shoulder", 90.0)
        el = angles.get("elbow",    160.0)
        gr = angles.get("gripper",  0.0)

    t1 = math.radians(sh)
    ta = t1 + (math.radians(el) - math.pi)

    # ── Posiciones de articulaciones ─────────────────────────────────
    p0 = (BASE_X, BASE_Y)
    p1 = (int(p0[0] + L1 * math.sin(t1)),
          int(p0[1] - L1 * math.cos(t1)))
    p2 = (int(p1[0] + L2 * math.sin(ta)),
          int(p1[1] - L2 * math.cos(ta)))
    p3 = (int(p2[0] + L3 * math.sin(ta)),
          int(p2[1] - L3 * math.cos(ta)))

    # ── Dirección del último eslabón (para orientar el gripper) ──────
    dx = p3[0] - p2[0]
    dy = p3[1] - p2[1]
    n  = math.sqrt(dx**2 + dy**2) + 1e-9
    # Vector unitario a lo largo del eslabón
    ux, uy = dx / n, dy / n
    # Vector perpendicular (para abrir la pinza)
    gap = int(L_GRIPPER * gr) + 8
    px, py = int(-uy * gap), int(ux * gap)

    # Raíces del gripper (donde nacen las dos ramas)
    g_root_top = (p3[0] + px, p3[1] + py)
    g_root_bot = (p3[0] - px, p3[1] - py)

    # Puntas del gripper: cada rama se dobla hacia adentro (hacia el eje)
    # Usamos el vector unitario del eslabón rotado ±30° hacia el centro
    angle_finger = math.radians(30)

    def rotate(vx, vy, a):
        return (vx * math.cos(a) - vy * math.sin(a),
                vx * math.sin(a) + vy * math.cos(a))

    # Rama superior: punta apunta hacia abajo-adelante
    fux_top, fuy_top = rotate(ux, uy, +angle_finger)
    g_tip_top = (int(g_root_top[0] + fux_top * L_FINGER),
                 int(g_root_top[1] + fuy_top * L_FINGER))

    # Rama inferior: punta apunta hacia arriba-adelante
    fux_bot, fuy_bot = rotate(ux, uy, -angle_finger)
    g_tip_bot = (int(g_root_bot[0] + fux_bot * L_FINGER),
                 int(g_root_bot[1] + fuy_bot * L_FINGER))

    # ── Fondo con cuadrícula suave ────────────────────────────────────
    for x in range(0, ROBOT_W, 40):
        cv2.line(canvas, (x, 0), (x, ROBOT_H), (20, 20, 20), 1)
    for y in range(0, ROBOT_H, 40):
        cv2.line(canvas, (0, y), (ROBOT_W, y), (20, 20, 20), 1)

    # ── Base ──────────────────────────────────────────────────────────
    cv2.rectangle(canvas,
        (p0[0] - 22, p0[1]),
        (p0[0] + 22, p0[1] + 14),
        (80, 80, 80), -1)
    cv2.rectangle(canvas,
        (p0[0] - 22, p0[1]),
        (p0[0] + 22, p0[1] + 14),
        (160, 160, 160), 1)
    # Tornillo central de la base
    cv2.circle(canvas, p0, 4, (180,180,180), -1)

    # ── Eslabones ─────────────────────────────────────────────────────
    cv2.line(canvas, p0, p1, (200, 120,  30), 6, cv2.LINE_AA)
    cv2.line(canvas, p1, p2, (200, 120,  30), 5, cv2.LINE_AA)
    cv2.line(canvas, p2, p3, (200, 120,  30), 4, cv2.LINE_AA)

    # Borde más claro sobre cada eslabón (efecto 3D simple)
    cv2.line(canvas, p0, p1, (255, 180,  60), 2, cv2.LINE_AA)
    cv2.line(canvas, p1, p2, (255, 180,  60), 2, cv2.LINE_AA)
    cv2.line(canvas, p2, p3, (255, 180,  60), 1, cv2.LINE_AA)

    # ── Gripper: ramas ────────────────────────────────────────────────
    cv2.line(canvas, p3, g_root_top, (0, 165, 255), 3, cv2.LINE_AA)
    cv2.line(canvas, p3, g_root_bot, (0, 165, 255), 3, cv2.LINE_AA)

    # ── Gripper: puntas (dedos) ───────────────────────────────────────
    cv2.line(canvas, g_root_top, g_tip_top, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, g_root_bot, g_tip_bot, (0, 200, 255), 2, cv2.LINE_AA)

    # Puntitos en las puntas
    cv2.circle(canvas, g_tip_top, 3, (0, 230, 255), -1)
    cv2.circle(canvas, g_tip_bot, 3, (0, 230, 255), -1)

    # ── Articulaciones ────────────────────────────────────────────────
    joints = [(p0, 8), (p1, 7), (p2, 6), (p3, 5)]
    for pt, r in joints:
        cv2.circle(canvas, pt, r,     (255, 255, 255), -1)
        cv2.circle(canvas, pt, r,     (200, 120,  30),  1)
        cv2.circle(canvas, pt, r - 3, (180, 100,  20), -1)

    # ── Etiquetas de articulaciones ───────────────────────────────────
    labels = {p0: "BASE", p1: "J1", p2: "J2", p3: "EF"}
    for pt, lbl in labels.items():
        cv2.putText(canvas, lbl,
            (pt[0] + 9, pt[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            (160, 160, 160), 1, cv2.LINE_AA)

    # ── Ángulos en las articulaciones ─────────────────────────────────
    cv2.putText(canvas, f"{sh:.0f}deg",
        (p1[0] - 34, p1[1] + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
        (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{el:.0f}deg",
        (p2[0] - 34, p2[1] + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
        (0, 220, 255), 1, cv2.LINE_AA)

    # ── Panel de estado (esquina superior) ───────────────────────────
    cv2.rectangle(canvas, (0, 0), (ROBOT_W, 58), (10, 10, 10), -1)
    cv2.line(canvas, (0, 58), (ROBOT_W, 58), (40, 40, 40), 1)

    cv2.putText(canvas, "BRAZO ROBOTICO VIRTUAL",
        (10, 18), cv2.FONT_HERSHEY_SIMPLEX,
        0.52, (255, 140, 0), 1, cv2.LINE_AA)

    gripper_state = f"Gripper: {gr*100:.0f}%  ({'ABIERTO' if gr > 0.5 else 'CERRADO'})"
    cv2.putText(canvas, f"Hombro:{sh:6.1f}  Codo:{el:6.1f}",
        (10, 36), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, gripper_state,
        (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (0, 165, 255), 1, cv2.LINE_AA)

    return canvas


def draw_hud(frame: np.ndarray, angles: dict | None, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 255, 136), 2, cv2.LINE_AA)

    if angles is None:
        cv2.putText(frame, "Sin deteccion",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return

    color = (0, 255, 0) if angles["valid"] else (0, 165, 255)
    cv2.putText(frame,
        f"Hombro:{angles['shoulder']:5.1f}  Codo:{angles['elbow']:5.1f}",
        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    hand_txt = (f"Gripper: {angles['gripper']*100:.1f}%"
                if angles["hand_detected"] else "Gripper: sin mano")
    cv2.putText(frame, hand_txt,
        (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
        0.52, (0, 165, 255), 1, cv2.LINE_AA)


def main() -> None:
    print("Iniciando pipeline M1+M2+M3+M4 ...")
    print("Presiona 'q' para salir.")

    # Posicionar ventanas lado a lado
    cv2.namedWindow("pose2robot - Camara")
    cv2.namedWindow("pose2robot - Robot")
    cv2.moveWindow("pose2robot - Camara", 0,   30)
    cv2.moveWindow("pose2robot - Robot",  660, 30)

    prev_time = time.time()

    with CaptureModule() as cam, PoseDetector() as detector:
        calculator = AngleCalculator()

        while True:
            # M1
            ok, bgr, rgb = cam.read()
            if not ok:
                break

            # M2
            result = detector.process(rgb)

            # M3
            angles = calculator.compute(result)
            if result is None:
                calculator.reset_buffers()

            # Ventana cámara con skeleton
            cam_frame = detector.draw_skeleton(bgr, result)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-9)
            prev_time = curr_time

            draw_hud(cam_frame, angles, fps)

            # M4 – ventana separada del robot
            robot_frame = draw_robot_window(angles)

            cv2.imshow("pose2robot - Camara", cam_frame)
            cv2.imshow("pose2robot - Robot",  robot_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Saliendo...")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()