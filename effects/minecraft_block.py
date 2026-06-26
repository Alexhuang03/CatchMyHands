"""
CatchMyHands - Minecraft Block Effect
======================================
Génère des textures 16x16 de blocs Minecraft (herbe, terre, pierre, eau, sable, brique, etc.)
et convertit la zone à l'intérieur du cadre en une mosaïque de blocs Minecraft 3D.
"""

import cv2
import numpy as np


class MinecraftBlockEffect:
    """
    Effet mosaïque Minecraft à l'intérieur du cadre de jeu.
    Calcule la couleur moyenne de chaque cellule du cadre, trouve la texture Minecraft
    la plus proche en termes de couleur BGR, et dessine le bloc correspondant.
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self._init_textures()

    def _init_textures(self):
        """Génère procéduralement des textures 16x16 style Minecraft."""
        size = self.block_size

        # ── 1. Terre (Dirt) ──
        dirt = np.zeros((size, size, 3), dtype=np.uint8)
        dirt[:, :] = [45, 80, 110]  # Marron moyen BGR
        np.random.seed(42)
        mask_dark = np.random.rand(size, size) < 0.25
        mask_light = np.random.rand(size, size) < 0.15
        dirt[mask_dark] = [30, 55, 80]
        dirt[mask_light] = [60, 100, 130]

        # ── 2. Herbe (Grass) ──
        grass = dirt.copy()
        grass[0:size // 4, :] = [35, 115, 35]  # Dessus vert
        for x in range(size):
            if x % 2 == 0:
                grass[size // 4, x] = [35, 115, 35]
            if x % 5 == 0:
                grass[size // 4 + 1, x] = [35, 115, 35]

        # ── 3. Pierre (Stone) ──
        stone = np.zeros((size, size, 3), dtype=np.uint8)
        stone[:, :] = [120, 120, 120]  # Gris
        np.random.seed(100)
        mask_dark = np.random.rand(size, size) < 0.25
        mask_light = np.random.rand(size, size) < 0.2
        stone[mask_dark] = [80, 80, 80]
        stone[mask_light] = [160, 160, 160]

        # ── 4. Planches de bois (Oak Planks) ──
        planks = np.zeros((size, size, 3), dtype=np.uint8)
        planks[:, :] = [70, 130, 180]  # Beige/bois clair
        # Lignes de planche
        for y in range(0, size, size // 4):
            planks[y, :] = [40, 90, 130]
        # Joints verticaux
        planks[0:size // 4, size // 4] = [40, 90, 130]
        planks[size // 4:size // 2, size * 3 // 4] = [40, 90, 130]
        planks[size // 2:size * 3 // 4, size // 8] = [40, 90, 130]
        planks[size * 3 // 4:, size * 5 // 8] = [40, 90, 130]

        # ── 5. Eau (Water) ──
        water = np.zeros((size, size, 3), dtype=np.uint8)
        water[:, :] = [180, 80, 20]  # Bleu BGR
        for y in range(size):
            for x in range(size):
                if (x + y) % (size // 2) == 0 or (x - y) % (size // 2) == 0:
                    water[y, x] = [210, 130, 60]

        # ── 6. Sable (Sand) ──
        sand = np.zeros((size, size, 3), dtype=np.uint8)
        sand[:, :] = [160, 220, 240]  # Jaune sable BGR
        np.random.seed(200)
        mask_dark = np.random.rand(size, size) < 0.15
        sand[mask_dark] = [140, 195, 210]

        # ── 7. Brique (Brick) ──
        brick = np.zeros((size, size, 3), dtype=np.uint8)
        brick[:, :] = [40, 50, 170]  # Rouge brique BGR
        for y in range(0, size, size // 3):
            brick[y, :] = [180, 180, 180]
        brick[0:size // 3, size // 2] = [180, 180, 180]
        brick[size // 3:size * 2 // 3, size // 4] = [180, 180, 180]
        brick[size // 3:size * 2 // 3, size * 3 // 4] = [180, 180, 180]
        brick[size * 2 // 3:, size // 2] = [180, 180, 180]

        # ── 8. Obsidienne (Obsidian) ──
        obsidian = np.zeros((size, size, 3), dtype=np.uint8)
        obsidian[:, :] = [40, 15, 40]  # Violet très foncé BGR
        np.random.seed(300)
        mask_purple = np.random.rand(size, size) < 0.15
        obsidian[mask_purple] = [100, 40, 100]

        # ── 9. Feuilles (Leaves) ──
        leaves = np.zeros((size, size, 3), dtype=np.uint8)
        leaves[:, :] = [10, 80, 15]  # Vert foncé BGR
        np.random.seed(500)
        mask_light = np.random.rand(size, size) < 0.3
        leaves[mask_light] = [20, 120, 25]

        # Stocker les blocs et leurs couleurs de référence
        self.blocks = [
            {"image": grass, "color": np.array([40, 120, 40], dtype=np.float32)},
            {"image": leaves, "color": np.array([15, 80, 10], dtype=np.float32)},
            {"image": dirt, "color": np.array([45, 80, 110], dtype=np.float32)},
            {"image": stone, "color": np.array([120, 120, 120], dtype=np.float32)},
            {"image": planks, "color": np.array([70, 130, 180], dtype=np.float32)},
            {"image": water, "color": np.array([180, 80, 20], dtype=np.float32)},
            {"image": sand, "color": np.array([160, 220, 240], dtype=np.float32)},
            {"image": brick, "color": np.array([40, 50, 170], dtype=np.float32)},
            {"image": obsidian, "color": np.array([40, 15, 40], dtype=np.float32)},
        ]

    def render(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """
        Remplace la région (x1, y1) -> (x2, y2) par une mosaïque de blocs Minecraft.
        """
        h, w = frame.shape[:2]

        # Clamper les coordonnées
        x1 = max(0, x1)
        x2 = min(w, x2)
        y1 = max(0, y1)
        y2 = min(h, y2)

        roi_w = x2 - x1
        roi_h = y2 - y1
        if roi_w <= self.block_size or roi_h <= self.block_size:
            return frame

        # Calculer le nombre de blocs
        cols = roi_w // self.block_size
        rows = roi_h // self.block_size

        grid_w = cols * self.block_size
        grid_h = rows * self.block_size

        # Centrer la grille de blocs dans le rectangle
        start_x = x1 + (roi_w - grid_w) // 2
        start_y = y1 + (roi_h - grid_h) // 2

        roi = frame[start_y:start_y + grid_h, start_x:start_x + grid_w]

        # Downscale de l'image pour obtenir l'exacte couleur moyenne de chaque bloc
        small_roi = cv2.resize(roi, (cols, rows), interpolation=cv2.INTER_AREA)

        # Remplacer chaque cellule de l'image par la texture Minecraft correspondante
        for r in range(rows):
            y_pos = start_y + r * self.block_size
            for c in range(cols):
                x_pos = start_x + c * self.block_size
                avg_color = small_roi[r, c].astype(np.float32)

                # Trouver la texture la plus proche
                best_dist = float('inf')
                best_texture = None
                for block in self.blocks:
                    # Distance euclidienne pondérée pour une meilleure correspondance visuelle
                    dist = np.sum((avg_color - block["color"]) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_texture = block["image"]

                if best_texture is not None:
                    frame[y_pos:y_pos + self.block_size, x_pos:x_pos + self.block_size] = best_texture

        return frame
