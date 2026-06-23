"""
CatchMyHands - Drawing Effect
=================================
Effet de dessin déclenché par le geste de pincement.
Trace des traits sur un canvas persistant entre les frames.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

import config


@dataclass
class DrawPoint:
    """Un point du trait avec ses métadonnées."""
    x: int                  # Position X en pixels
    y: int                  # Position Y en pixels
    thickness: int          # Épaisseur du trait
    age: int = 0            # Âge en frames (pour le fade)
    is_break: bool = False  # True = interruption du trait (lever le stylo)


class DrawingEffect:
    """
    Moteur de dessin par pincement.

    Trace des lignes continues entre les positions de pincement consécutives.
    Gère le fade progressif, l'épaisseur dynamique basée sur la vitesse,
    et l'effacement par geste.
    """

    def __init__(self, width: int, height: int):
        """
        Args:
            width: Largeur du canvas en pixels.
            height: Hauteur du canvas en pixels.
        """
        self.width = width
        self.height = height

        # Buffer de points par main
        self._trails: dict[int, List[DrawPoint]] = {}
        # Position précédente pour calculer la vitesse
        self._prev_pos: dict[int, Tuple[int, int]] = {}
        # Canvas persistant (overlay transparent)
        self._canvas = np.zeros((height, width, 4), dtype=np.uint8)
        self._canvas_dirty = True

    def update(self, hand_index: int, pinch_pos: Tuple[float, float],
               is_pinching: bool):
        """
        Met à jour l'état du dessin pour une main.

        Args:
            hand_index: Index de la main.
            pinch_pos: Position normalisée (x, y) du pincement.
            is_pinching: True si le geste de pincement est actif.
        """
        # Convertir coordonnées normalisées → pixels
        px = int(pinch_pos[0] * self.width)
        py = int(pinch_pos[1] * self.height)

        # Clamp dans le cadre
        px = max(0, min(self.width - 1, px))
        py = max(0, min(self.height - 1, py))

        if hand_index not in self._trails:
            self._trails[hand_index] = []

        if is_pinching:
            # Calculer l'épaisseur basée sur la vitesse
            thickness = self._compute_thickness(hand_index, px, py)

            # Ajouter un point au trail
            self._trails[hand_index].append(DrawPoint(
                x=px, y=py, thickness=thickness
            ))
            self._prev_pos[hand_index] = (px, py)
            self._canvas_dirty = True
        else:
            # Pincement relâché → ajouter un point de rupture
            if (hand_index in self._trails and
                    len(self._trails[hand_index]) > 0 and
                    not self._trails[hand_index][-1].is_break):
                self._trails[hand_index].append(DrawPoint(
                    x=px, y=py, is_break=True
                ))
            # Reset la position précédente
            self._prev_pos.pop(hand_index, None)

    def _compute_thickness(self, hand_index: int, px: int, py: int) -> int:
        """
        Calcule l'épaisseur du trait inversement proportionnelle à la vitesse.
        Mouvement lent → trait épais / Mouvement rapide → trait fin.
        """
        if hand_index not in self._prev_pos:
            return config.DRAW_THICKNESS_MAX

        prev_x, prev_y = self._prev_pos[hand_index]
        speed = math.sqrt((px - prev_x) ** 2 + (py - prev_y) ** 2)

        # Mapper la vitesse sur l'épaisseur (inversé)
        max_speed = 50.0  # pixels/frame
        ratio = min(1.0, speed / max_speed)

        thickness = int(
            config.DRAW_THICKNESS_MAX -
            ratio * (config.DRAW_THICKNESS_MAX - config.DRAW_THICKNESS_MIN)
        )
        return max(config.DRAW_THICKNESS_MIN, thickness)

    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Rend les traits de dessin sur le frame.

        Args:
            frame: Frame BGR sur lequel dessiner.

        Returns:
            Frame avec les traits dessinés.
        """
        if self._canvas_dirty:
            self._rebuild_canvas()
            self._canvas_dirty = False

        # Alpha blending du canvas sur le frame
        if np.any(self._canvas[:, :, 3] > 0):
            alpha = self._canvas[:, :, 3:4].astype(np.float32) / 255.0
            canvas_bgr = self._canvas[:, :, :3].astype(np.float32)
            frame_float = frame.astype(np.float32)

            blended = (alpha * canvas_bgr + (1.0 - alpha) * frame_float)
            frame = blended.astype(np.uint8)

        return frame

    def _rebuild_canvas(self):
        """Reconstruit le canvas à partir de tous les trails."""
        self._canvas[:] = 0

        for hand_index, trail in self._trails.items():
            if len(trail) < 2:
                continue

            for i in range(1, len(trail)):
                if trail[i].is_break or trail[i - 1].is_break:
                    continue

                p1 = (trail[i - 1].x, trail[i - 1].y)
                p2 = (trail[i].x, trail[i].y)
                thickness = trail[i].thickness

                # Calculer l'opacité (fade si activé)
                if config.DRAW_FADE_ENABLED:
                    age = len(trail) - i
                    alpha = max(0, 255 - int(255 * age / config.DRAW_FADE_DURATION))
                else:
                    alpha = 255

                if alpha <= 0:
                    continue

                color_with_alpha = (
                    config.DRAW_COLOR[0],
                    config.DRAW_COLOR[1],
                    config.DRAW_COLOR[2],
                    alpha,
                )

                # Dessiner la ligne sur le canvas BGRA
                cv2.line(self._canvas, p1, p2, color_with_alpha, thickness,
                         lineType=cv2.LINE_AA)

    def clear(self):
        """Efface complètement le canvas."""
        self._trails.clear()
        self._prev_pos.clear()
        self._canvas[:] = 0
        self._canvas_dirty = False
        print("[DrawingEffect] Canvas effacé")
