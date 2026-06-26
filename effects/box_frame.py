"""
CatchMyHands - Box Frame Effect
==================================
Effet de cadre néon interactif dessiné entre les deux mains (pouces et index).
Affiche un polygone translucide avec des bordures néon et des réticules de visée.
"""

import cv2
import numpy as np
import config
from effects.minecraft_effect import MinecraftEffect

class BoxFrameEffect:
    """
    Gère le rendu visuel du cadre interactif lorsque les index et pouces
    des deux mains se touchent pour former un cadre.
    """

    def __init__(self):
        self.last_box = None  # (x1, y1, x2, y2) de la dernière frame
        self.minecraft_pixelate = MinecraftEffect()

    def close(self):
        """Libère les ressources de l'effet de cadre."""
        pass

    def render(self, frame: np.ndarray, left_landmarks: np.ndarray, right_landmarks: np.ndarray, bw_filter: bool = False, pix_filter: bool = False) -> np.ndarray:
        """
        Dessine le cadre néon et les effets visuels associés sur l'image.

        Args:
            frame: L'image BGR d'OpenCV.
            left_landmarks: Landmarks de la main gauche (21, 3).
            right_landmarks: Landmarks de la main droite (21, 3).
            bw_filter: Si True, convertit l'intérieur du cadre en noir et blanc.
            pix_filter: Si True, convertit l'intérieur du cadre en pixelisation simple.

        Returns:
            L'image avec l'effet appliqué.
        """
        h, w = frame.shape[:2]

        # Les 4 sommets du polygone définis par les index et les pouces
        pts = np.array([
            (int(left_landmarks[8][0] * w), int(left_landmarks[8][1] * h)),   # Top-Left (Index Gauche)
            (int(right_landmarks[8][0] * w), int(right_landmarks[8][1] * h)), # Top-Right (Index Droit)
            (int(right_landmarks[4][0] * w), int(right_landmarks[4][1] * h)), # Bottom-Right (Pouce Droit)
            (int(left_landmarks[4][0] * w), int(left_landmarks[4][1] * h))    # Bottom-Left (Pouce Gauche)
        ], dtype=np.int32)

        # Calculer la bounding box pour l'effet de bris de glace et les opérations de filtrage
        x1 = max(0, int(np.min(pts[:, 0])))
        x2 = min(w, int(np.max(pts[:, 0])))
        y1 = max(0, int(np.min(pts[:, 1])))
        y2 = min(h, int(np.max(pts[:, 1])))

        self.last_box = (x1, y1, x2, y2)

        # Appliquer les filtres dans le polygone
        if (bw_filter or pix_filter) and (x2 > x1 and y2 > y1):
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            roi_mask = mask[y1:y2, x1:x2]
            roi = frame[y1:y2, x1:x2]

            if bw_filter:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                filtered_roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                frame[y1:y2, x1:x2] = np.where(roi_mask[:, :, None] == 255, filtered_roi, roi)

            elif pix_filter:
                filtered_roi = self.minecraft_pixelate.render(roi)
                frame[y1:y2, x1:x2] = np.where(roi_mask[:, :, None] == 255, filtered_roi, roi)

        # ── 1. Remplissage semi-transparent (Polygone) ──
        # On n'applique pas de teinte de couleur cyan si un filtre est actif (B&W ou Pixelate)
        if not bw_filter and not pix_filter:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], config.FRAME_COLOR)
            cv2.addWeighted(overlay, config.FRAME_FILL_OPACITY, frame, 1.0 - config.FRAME_FILL_OPACITY, 0, frame)

        # ── 2. Lignes de bordure néon (Effet de lueur) ──
        # Lueur externe (largeur 6, couleur d'origine avec lissage LINE_AA)
        cv2.polylines(frame, [pts], isClosed=True, color=config.FRAME_COLOR, thickness=6, lineType=cv2.LINE_AA)
        
        # Noyau interne brillant (largeur 2, couleur presque blanche pour simuler du néon)
        neon_white = (
            min(255, config.FRAME_COLOR[0] + 50),
            min(255, config.FRAME_COLOR[1] + 50),
            min(255, config.FRAME_COLOR[2] + 50)
        )
        cv2.polylines(frame, [pts], isClosed=True, color=neon_white, thickness=2, lineType=cv2.LINE_AA)

        # ── 3. Dessiner des réticules/coins de visée aux 4 sommets ──
        for pt in [
            (int(left_landmarks[8][0] * w), int(left_landmarks[8][1] * h)),
            (int(right_landmarks[8][0] * w), int(right_landmarks[8][1] * h)),
            (int(right_landmarks[4][0] * w), int(right_landmarks[4][1] * h)),
            (int(left_landmarks[4][0] * w), int(left_landmarks[4][1] * h))
        ]:
            self._draw_reticle(frame, pt)


        return frame

    def _draw_reticle(self, frame: np.ndarray, pt: tuple):
        """Dessine un petit réticule de visée futuriste autour d'un point."""
        x, y = pt
        size = 10
        color = config.FRAME_COLOR
        thickness = 1

        # Cercle central
        cv2.circle(frame, (x, y), 3, color, -1, lineType=cv2.LINE_AA)

        # Lignes en croix
        cv2.line(frame, (x - size, y), (x - 4, y), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(frame, (x + size, y), (x + 4, y), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(frame, (x, y - size), (x, y - 4), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(frame, (x, y + size), (x, y + 4), color, thickness, lineType=cv2.LINE_AA)
