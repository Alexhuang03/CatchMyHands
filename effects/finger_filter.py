"""
CatchMyHands - Finger Filter Effect
======================================
Applique des filtres visuels dans des cadres polygonaux
soit en mode mono-main (mini-cadres), soit en mode bi-manuel (cadres étirés entre les mains).
Supporte jusqu'à 4 filtres cyberpunk.
"""

import cv2
import numpy as np
import config
from effects.minecraft_effect import MinecraftEffect


class FingerFilterEffect:
    """
    Gère le rendu des cadres de filtre par doigt.
    Supporte le mode une seule main ou les cadres géants étirés entre deux mains.
    """

    # Mapping des doigts vers les landmarks pour le mode une main (pouce ↔ articulation/bout)
    FINGER_LANDMARKS = {
        0: (4, 3, 6, 8),    # Index:      Thumb tip, Thumb IP, Index PIP, Index tip
        1: (4, 3, 10, 12),   # Majeur:     Thumb tip, Thumb IP, Middle PIP, Middle tip
        2: (4, 3, 14, 16),   # Annulaire:  Thumb tip, Thumb IP, Ring PIP, Ring tip
        3: (4, 3, 18, 20),   # Auriculaire: Thumb tip, Thumb IP, Pinky PIP, Pinky tip
    }

    # Bout de doigt spécifique pour le mode deux mains
    FINGER_TIPS = [8, 12, 16, 20]

    FINGER_KEYS = ["index", "middle", "ring", "pinky"]

    def __init__(self):
        """Initialise l'effet de filtre."""
        self.minecraft_pixelate = MinecraftEffect()

    def render_single_hand(self, frame: np.ndarray, landmarks: np.ndarray,
                           active_filters_ordered: list) -> np.ndarray:
        """
        Applique les filtres dans les mini-cadres d'une seule main.
        """
        if landmarks is None or len(landmarks) < 21:
            return frame

        h, w = frame.shape[:2]

        for finger_idx in range(4):
            if finger_idx >= len(active_filters_ordered):
                continue

            filter_name = active_filters_ordered[finger_idx]
            finger_key = self.FINGER_KEYS[finger_idx]
            color = config.FINGER_FILTER_COLORS[finger_key]

            # Construire le polygone du mini-cadre
            lm_indices = self.FINGER_LANDMARKS[finger_idx]
            pts = np.array([
                (int(landmarks[idx][0] * w), int(landmarks[idx][1] * h))
                for idx in lm_indices
            ], dtype=np.int32)

            frame = self._draw_and_filter_polygon(frame, pts, filter_name, color)

        return frame

    def render_two_hands(self, frame: np.ndarray, lm_left: np.ndarray, lm_right: np.ndarray,
                         active_filters_ordered: list) -> np.ndarray:
        """
        Applique les filtres dans les cadres bi-manuels étirés entre les deux mains.
        Les coins de chaque cadre sont :
        - Index/doigt gauche
        - Pouce gauche
        - Pouce droit
        - Index/doigt droit
        """
        if lm_left is None or lm_right is None:
            return frame

        h, w = frame.shape[:2]

        for finger_idx in range(4):
            if finger_idx >= len(active_filters_ordered):
                continue

            filter_name = active_filters_ordered[finger_idx]
            finger_key = self.FINGER_KEYS[finger_idx]
            color = config.FINGER_FILTER_COLORS[finger_key]

            tip_idx = self.FINGER_TIPS[finger_idx]

            # Définir les 4 sommets du cadre bi-manuel :
            # 1. Doigt gauche (tip)
            # 2. Pouce gauche (tip)
            # 3. Pouce droit (tip)
            # 4. Doigt droit (tip)
            pts = np.array([
                (int(lm_left[tip_idx][0] * w), int(lm_left[tip_idx][1] * h)),
                (int(lm_left[4][0] * w), int(lm_left[4][1] * h)),
                (int(lm_right[4][0] * w), int(lm_right[4][1] * h)),
                (int(lm_right[tip_idx][0] * w), int(lm_right[tip_idx][1] * h))
            ], dtype=np.int32)

            frame = self._draw_and_filter_polygon(frame, pts, filter_name, color)

        return frame

    def _draw_and_filter_polygon(self, frame: np.ndarray, pts: np.ndarray,
                                 filter_name: str, color: tuple) -> np.ndarray:
        """
        Trie le polygone, applique le filtre et dessine la bordure néon.
        """
        h, w = frame.shape[:2]

        # Trier les points par angle polaire pour éviter les croisements
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        pts = pts[np.argsort(angles)]

        # Bounding box pour le ROI
        x1 = max(0, int(np.min(pts[:, 0])))
        x2 = min(w, int(np.max(pts[:, 0])))
        y1 = max(0, int(np.min(pts[:, 1])))
        y2 = min(h, int(np.max(pts[:, 1])))

        if x2 > x1 and y2 > y1:
            # ── Appliquer le filtre dans le polygone ──
            frame = self._apply_filter_in_polygon(
                frame, pts, x1, y1, x2, y2, filter_name, color
            )

        # ── Bordure néon du cadre ──
        # Lueur externe
        cv2.polylines(frame, [pts], isClosed=True, color=color,
                      thickness=config.FINGER_FILTER_GLOW_THICKNESS,
                      lineType=cv2.LINE_AA)
        # Noyau interne brillant
        neon_white = (
            min(255, color[0] + 60),
            min(255, color[1] + 60),
            min(255, color[2] + 60)
        )
        cv2.polylines(frame, [pts], isClosed=True, color=neon_white,
                      thickness=config.FINGER_FILTER_BORDER_THICKNESS,
                      lineType=cv2.LINE_AA)

        # Petit point cible/croix sur chaque sommet du polygone
        for pt in pts:
            cv2.circle(frame, (pt[0], pt[1]), 3, neon_white, -1, lineType=cv2.LINE_AA)

        return frame

    def _apply_filter_in_polygon(self, frame: np.ndarray, pts: np.ndarray,
                                  x1: int, y1: int, x2: int, y2: int,
                                  filter_name: str, color: tuple) -> np.ndarray:
        """
        Applique un filtre nommé à l'intérieur d'un polygone.
        """
        # Créer le masque du polygone
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        roi_mask = mask[y1:y2, x1:x2]
        roi = frame[y1:y2, x1:x2]

        if filter_name == "bw":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            filtered_roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif filter_name == "pixelate":
            filtered_roi = self.minecraft_pixelate.render(roi)
        elif filter_name == "invert":
            filtered_roi = cv2.bitwise_not(roi)
        elif filter_name == "edge":
            # Filtre néon cyber-contour
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Filtre médian pour enlever du bruit
            blurred = cv2.medianBlur(gray, 3)
            # Canny contours
            edges = cv2.Canny(blurred, 30, 90)
            
            # Créer un masque de couleur pour les contours
            neon_contour = np.zeros_like(roi)
            neon_contour[:] = color
            
            # Dessiner les contours néons sur l'original
            filtered_roi = np.where(edges[:, :, None] == 255, neon_contour, roi)
            # Ajouter une touche sombre pour faire ressortir l'effet
            darkened = cv2.addWeighted(filtered_roi, 0.7, np.zeros_like(roi), 0.3, 0)
            filtered_roi = np.where(edges[:, :, None] == 255, neon_contour, darkened)
        else:
            return frame

        # Appliquer le filtre uniquement dans le masque du polygone
        frame[y1:y2, x1:x2] = np.where(
            roi_mask[:, :, None] == 255, filtered_roi, roi
        )

        return frame
