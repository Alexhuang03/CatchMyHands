"""
CatchMyHands - Minecraft Face Mask Effect
==========================================
Détecte le visage à l'intérieur du cadre et dessine la tête de Steve (Minecraft)
par-dessus avec lissage temporel (EMA) pour éviter le sautillement.
"""

import cv2
import numpy as np


class MinecraftFaceEffect:
    """
    Détecte le visage de l'utilisateur à l'intérieur du cadre
    et applique un masque de tête de Steve Minecraft en temps réel.
    """

    def __init__(self):
        # Charger le classificateur de visage OpenCV Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Générer la texture de la tête de Steve (8x8 pixels)
        self.steve_face = self._generate_steve_face()
        
        # Variables de lissage (EMA) pour la boîte du visage
        self._smoothed_face = None  # [x, y, w, h]
        self._lost_frames = 0
        self._MAX_LOST_FRAMES = 8    # Nombre de frames de tolérance en cas de perte de détection
        self._smooth_alpha = 0.35    # Facteur EMA (plus bas = plus fluide, plus haut = réactif)

    def _generate_steve_face(self) -> np.ndarray:
        """Génère la texture couleur officielle de la tête de Steve Minecraft."""
        face = np.zeros((8, 8, 3), dtype=np.uint8)
        
        # Palette de couleurs Minecraft Steve (format BGR)
        HAIR = [12, 34, 62]     # #3e220c - Marron foncé
        SKIN = [125, 172, 226]  # #e2ac7d - Peau pêche
        WHITE = [255, 255, 255] # Blanc des yeux
        BLUE = [215, 115, 60]   # #3c73d7 - Yeux bleus Steve
        NOSE = [90, 115, 170]   # #aa735a - Peau plus foncée (nez)
        BEARD = [18, 38, 68]    # #442612 - Barbe/Moustache
        MOUTH = [35, 45, 115]   # #732d23 - Lèvres rouges
        
        # Lignes 0 & 1 : Cheveux
        face[0, :] = HAIR
        face[1, :] = HAIR
        
        # Ligne 2 : Cheveux sur les côtés, peau au milieu
        face[2, 0] = HAIR
        face[2, 1:7] = SKIN
        face[2, 7] = HAIR
        
        # Ligne 3 : Peau complète
        face[3, :] = SKIN
        
        # Ligne 4 : Yeux (Blanc/Bleu) et peau
        face[4, 0] = SKIN
        face[4, 1] = WHITE
        face[4, 2] = BLUE
        face[4, 3:5] = SKIN
        face[4, 5] = BLUE
        face[4, 6] = WHITE
        face[4, 7] = SKIN
        
        # Ligne 5 : Nez
        face[5, 0:3] = SKIN
        face[5, 3:5] = NOSE
        face[5, 5:8] = SKIN
        
        # Ligne 6 : Bouche, moustache et peau
        face[6, 0] = SKIN
        face[6, 1] = BEARD
        face[6, 2] = SKIN
        face[6, 3:5] = MOUTH
        face[6, 5] = SKIN
        face[6, 6] = BEARD
        face[6, 7] = SKIN
        
        # Ligne 7 : Menton/barbe
        face[7, 0] = SKIN
        face[7, 1:7] = BEARD
        face[7, 7] = SKIN
        
        return face

    def render(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """
        Détecte le visage dans la zone du cadre et dessine la tête de Steve par-dessus.
        """
        h, w = frame.shape[:2]

        # Clamper les coordonnées de la zone de rendu
        x1 = max(0, x1)
        x2 = min(w, x2)
        y1 = max(0, y1)
        y2 = min(h, y2)

        roi_w = x2 - x1
        roi_h = y2 - y1
        # Si le cadre est trop petit, inutile de chercher un visage
        if roi_w < 60 or roi_h < 60:
            return frame

        # Extraire la zone d'intérêt
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Détecter le visage dans la zone du cadre
        # On définit une taille minimale dynamique pour éviter les faux positifs de bruit de fond
        min_size = int(min(roi_w, roi_h) * 0.2)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_size, min_size)
        )

        target_face = None

        if len(faces) > 0:
            # Garder le visage le plus grand (le plus proche)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            target_face = faces[0]
            self._lost_frames = 0
        else:
            # En cas de perte de détection sur quelques frames, on garde la position précédente
            self._lost_frames += 1
            if self._lost_frames < self._MAX_LOST_FRAMES:
                target_face = self._smoothed_face

        if target_face is not None:
            # Lissage EMA des coordonnées [x, y, w, h] pour éviter le flickering
            if self._smoothed_face is None:
                self._smoothed_face = np.array(target_face, dtype=np.float32)
            else:
                self._smoothed_face = (
                    self._smoothed_face * (1.0 - self._smooth_alpha) +
                    np.array(target_face, dtype=np.float32) * self._smooth_alpha
                )

            # Coordonnées entières lissées
            fx, fy, fw, fh = self._smoothed_face.astype(int)

            # Élargir la boîte pour couvrir les cheveux, les oreilles et le menton (aspect cubique)
            pad_w = int(fw * 0.15)
            pad_h = int(fh * 0.15)

            # Coordonnées globales dans le frame
            sx1 = x1 + fx - pad_w
            sy1 = y1 + fy - pad_h
            sx2 = x1 + fx + fw + pad_w
            sy2 = y1 + fy + fh + pad_h

            # Clamper pour l'affichage (permet à Steve de dépasser du cadre de manière naturelle)
            sx1_c = max(0, sx1)
            sy1_c = max(0, sy1)
            sx2_c = min(w, sx2)
            sy2_c = min(h, sy2)

            overlap_w = sx2_c - sx1_c
            overlap_h = sy2_c - sy1_c

            if overlap_w > 0 and overlap_h > 0:
                # Recréer la tête à la taille complète
                target_w = sx2 - sx1
                target_h = sy2 - sy1
                steve_resized = cv2.resize(self.steve_face, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

                # Extraire la partie visible
                crop_x1 = sx1_c - sx1
                crop_y1 = sy1_c - sy1
                crop_x2 = crop_x1 + overlap_w
                crop_y2 = crop_y1 + overlap_h

                steve_visible = steve_resized[crop_y1:crop_y2, crop_x1:crop_x2]

                # Remplacer les pixels de l'image réelle par Steve
                frame[sy1_c:sy2_c, sx1_c:sx2_c] = steve_visible
        else:
            self._smoothed_face = None

        return frame
