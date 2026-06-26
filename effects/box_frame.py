"""
CatchMyHands - Box Frame Effect
==================================
Effet de cadre néon interactif dessiné entre les deux mains (pouces et index).
Affiche un polygone translucide avec des bordures néon et des réticules de visée.
"""

import cv2
import numpy as np
import config
from effects.minecraft_block import MinecraftFaceEffect
from effects.minecraft_effect import MinecraftEffect

class BoxFrameEffect:
    """
    Gère le rendu visuel du cadre interactif lorsque les index et pouces
    des deux mains se touchent pour former un cadre.
    """

    def __init__(self):
        self.last_box = None  # (x1, y1, x2, y2) de la dernière frame
        self.minecraft_face = MinecraftFaceEffect()
        self.minecraft_pixelate = MinecraftEffect()

    def render(self, frame: np.ndarray, left_landmarks: np.ndarray, right_landmarks: np.ndarray, bw_filter: bool = False, mc_filter: bool = False, pix_filter: bool = False) -> np.ndarray:
        """
        Dessine le cadre néon et les effets visuels associés sur l'image.

        Args:
            frame: L'image BGR d'OpenCV.
            left_landmarks: Landmarks de la main gauche (21, 3).
            right_landmarks: Landmarks de la main droite (21, 3).
            bw_filter: Si True, convertit l'intérieur du cadre en noir et blanc.
            mc_filter: Si True, convertit l'intérieur du cadre en mosaïque Minecraft.
            pix_filter: Si True, convertit l'intérieur du cadre en pixelisation simple.

        Returns:
            L'image avec l'effet appliqué.
        """
        h, w = frame.shape[:2]

        # Calculer le point de pincement pour chaque main (milieu entre pouce et index)
        l_px = int((left_landmarks[4][0] + left_landmarks[8][0]) / 2.0 * w)
        l_py = int((left_landmarks[4][1] + left_landmarks[8][1]) / 2.0 * h)
        r_px = int((right_landmarks[4][0] + right_landmarks[8][0]) / 2.0 * w)
        r_py = int((right_landmarks[4][1] + right_landmarks[8][1]) / 2.0 * h)

        # Les 4 sommets du rectangle aligné sur les axes
        pts = np.array([
            (l_px, l_py),
            (r_px, l_py),
            (r_px, r_py),
            (l_px, r_py)
        ], dtype=np.int32)

        # Sauvegarder la bounding box pour l'effet de bris de glace
        self.last_box = (min(l_px, r_px), min(l_py, r_py), max(l_px, r_px), max(l_py, r_py))

        # ── Option 1 : Filtre Noir et Blanc à l'intérieur du cadre ──
        if bw_filter:
            x1 = max(0, min(l_px, r_px))
            x2 = min(w, max(l_px, r_px))
            y1 = max(0, min(l_py, r_py))
            y2 = min(h, max(l_py, r_py))
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                frame[y1:y2, x1:x2] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # ── Option 2 : Filtre Mosaïque Minecraft à l'intérieur du cadre ──
        if mc_filter:
            x1 = max(0, min(l_px, r_px))
            x2 = min(w, max(l_px, r_px))
            y1 = max(0, min(l_py, r_py))
            y2 = min(h, max(l_py, r_py))
            if x2 > x1 and y2 > y1:
                frame = self.minecraft_face.render(frame, x1, y1, x2, y2)

        # ── Option 3 : Filtre Pixelate à l'intérieur du cadre ──
        if pix_filter:
            x1 = max(0, min(l_px, r_px))
            x2 = min(w, max(l_px, r_px))
            y1 = max(0, min(l_py, r_py))
            y2 = min(h, max(l_py, r_py))
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                frame[y1:y2, x1:x2] = self.minecraft_pixelate.render(roi)

        # ── 1. Remplissage semi-transparent (Polygone) ──
        # On n'applique pas de teinte de couleur cyan si un filtre est actif (B&W, Minecraft ou Pixelate)
        if not bw_filter and not mc_filter and not pix_filter:
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
        for pt in [(l_px, l_py), (r_px, l_py), (r_px, r_py), (l_px, r_py)]:
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
