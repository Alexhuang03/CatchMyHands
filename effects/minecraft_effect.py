"""
CatchMyHands - Minecraft Effect
================================
Effet de pixelisation style Minecraft appliqué sur tout le frame.
Transforme la vidéo en blocs carrés avec grille optionnelle
et réduction de palette pour un look jeu vidéo rétro.
"""

import cv2
import numpy as np

import config


class MinecraftEffect:
    """
    Applique un effet de pixelisation style Minecraft sur le frame.
    
    Technique : downscale avec INTER_NEAREST → upscale INTER_NEAREST.
    Très léger (< 1ms par frame) car ce sont juste 2 resize.
    """

    def __init__(self):
        """Initialise l'effet Minecraft."""
        pass

    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Applique l'effet Minecraft sur le frame entier.

        Args:
            frame: Image BGR d'OpenCV.

        Returns:
            L'image pixelisée style Minecraft.
        """
        h, w = frame.shape[:2]
        block = config.MINECRAFT_BLOCK_SIZE

        # Dimensions réduites (nombre de blocs)
        small_w = max(1, w // block)
        small_h = max(1, h // block)

        # ── 1. Downscale → Upscale (pixelisation) ──
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

        # ── 2. Réduction de palette (optionnel) ──
        if config.MINECRAFT_COLOR_REDUCE:
            # Réduire chaque canal à ~32 niveaux (8 bits → 5 bits effectifs)
            # Cela donne un look plus "jeu vidéo" avec des couleurs franches
            divisor = 8
            small = (small // divisor) * divisor + divisor // 2

        # ── 3. Upscale à la taille originale ──
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # ── 4. Grille entre les blocs (optionnel) ──
        if config.MINECRAFT_GRID_LINES:
            # Lignes verticales
            for x in range(0, w, block):
                cv2.line(pixelated, (x, 0), (x, h), (20, 20, 20), 1)
            # Lignes horizontales
            for y in range(0, h, block):
                cv2.line(pixelated, (0, y), (w, y), (20, 20, 20), 1)

        return pixelated
