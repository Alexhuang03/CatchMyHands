"""
CatchMyHands - Overlay Effect
=================================
Effet de superposition d'image (aura) déclenché par la main ouverte.
Gère le positionnement, le redimensionnement dynamique, la pulsation
et l'alpha blending.
"""

import math
import os

import cv2
import numpy as np

import config


class OverlayEffect:
    """
    Superpose une image PNG (avec canal alpha) sur le flux vidéo.

    Positionné au centre de la paume quand la main est ouverte,
    avec animation de pulsation et redimensionnement proportionnel
    à la taille de la main.
    """

    def __init__(self):
        """Charge l'image overlay depuis le fichier asset."""
        self._overlay = None
        self._pulse_phase = 0.0
        self._load_overlay()

    def _load_overlay(self):
        """Charge l'image PNG avec canal alpha."""
        if not os.path.exists(config.AURA_IMAGE_PATH):
            print(f"[OverlayEffect] ⚠ Image non trouvée : {config.AURA_IMAGE_PATH}")
            print("[OverlayEffect] Génération d'un overlay procédural...")
            self._overlay = self._generate_procedural_aura(256, 256)
            return

        self._overlay = cv2.imread(config.AURA_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

        if self._overlay is None:
            print(f"[OverlayEffect] ⚠ Impossible de charger : {config.AURA_IMAGE_PATH}")
            self._overlay = self._generate_procedural_aura(256, 256)
            return

        # S'assurer que l'image a un canal alpha
        if self._overlay.shape[2] == 3:
            alpha = np.full(
                (self._overlay.shape[0], self._overlay.shape[1], 1),
                255, dtype=np.uint8
            )
            self._overlay = np.concatenate([self._overlay, alpha], axis=2)

        print(f"[OverlayEffect] Image chargée : {self._overlay.shape}")

    def _generate_procedural_aura(self, width: int, height: int) -> np.ndarray:
        """
        Génère une aura procédurale (gradient circulaire avec glow).
        Utilisé comme fallback si l'image PNG n'est pas disponible.
        """
        aura = np.zeros((height, width, 4), dtype=np.uint8)
        cx, cy = width // 2, height // 2
        max_radius = min(cx, cy)

        for y in range(height):
            for x in range(width):
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist > max_radius:
                    continue

                # Gradient circulaire
                ratio = dist / max_radius
                intensity = int(255 * (1.0 - ratio) ** 1.5)

                # Couleur : dégradé cyan → bleu → violet
                r = int(80 + 175 * ratio)
                g = int(200 * (1.0 - ratio ** 2))
                b = 255

                aura[y, x] = [b, g, r, intensity]

        return aura

    def render(self, frame: np.ndarray, palm_center: tuple,
               hand_size: float, edge_factor: float = 1.0) -> np.ndarray:
        """
        Rend l'overlay aura sur le frame.

        Args:
            frame: Frame BGR.
            palm_center: Centre de la paume (x, y) normalisé [0, 1].
            hand_size: Taille de la main normalisée.
            edge_factor: Facteur d'atténuation aux bords [0, 1].

        Returns:
            Frame avec l'overlay appliqué.
        """
        if self._overlay is None:
            return frame

        h, w = frame.shape[:2]

        # ── Position en pixels ──
        cx = int(palm_center[0] * w)
        cy = int(palm_center[1] * h)

        # ── Taille de l'overlay ──
        base_size = int(hand_size * w * config.AURA_SCALE_FACTOR)

        # Animation de pulsation
        self._pulse_phase += config.AURA_PULSE_SPEED
        pulse = 1.0 + config.AURA_PULSE_AMPLITUDE * math.sin(self._pulse_phase)
        target_size = int(base_size * pulse)

        if target_size < 10:
            return frame

        # ── Redimensionner l'overlay ──
        resized = cv2.resize(self._overlay, (target_size, target_size),
                             interpolation=cv2.INTER_AREA)

        # ── Calculer la ROI (Region of Interest) ──
        x1 = cx - target_size // 2
        y1 = cy - target_size // 2
        x2 = x1 + target_size
        y2 = y1 + target_size

        # Clamp aux bords du frame
        ox1 = max(0, -x1)  # Offset dans l'overlay
        oy1 = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        if x2 <= x1 or y2 <= y1:
            return frame

        # ── Alpha blending ──
        overlay_roi = resized[oy1:oy2, ox1:ox2]
        if overlay_roi.shape[0] == 0 or overlay_roi.shape[1] == 0:
            return frame

        alpha = overlay_roi[:, :, 3:4].astype(np.float32) / 255.0
        alpha *= config.AURA_BASE_OPACITY * edge_factor

        overlay_bgr = overlay_roi[:, :, :3].astype(np.float32)
        frame_roi = frame[y1:y2, x1:x2].astype(np.float32)

        blended = alpha * overlay_bgr + (1.0 - alpha) * frame_roi
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)

        return frame
