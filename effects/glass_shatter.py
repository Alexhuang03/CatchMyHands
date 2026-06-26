"""
CatchMyHands - Glass Shatter Effect (Lightweight)
====================================================
Animation de bris de glace légère et fluide.
Dessine directement sur le frame sans copies ni blending coûteux.
"""

import cv2
import numpy as np
import random
import math


class GlassShard:
    """Fragment de verre léger."""

    __slots__ = ('cx', 'cy', 'vx', 'vy', 'size', 'angle', 'angular_vel',
                 'color', 'opacity', 'age', 'max_age', 'shape')

    def __init__(self, cx, cy, vx, vy, size, color, shape):
        self.cx = cx
        self.cy = cy
        self.vx = vx
        self.vy = vy
        self.size = size
        self.angle = random.uniform(0, 6.28)
        self.angular_vel = random.uniform(-0.15, 0.15)
        self.color = color
        self.opacity = 1.0
        self.age = 0
        self.max_age = random.randint(28, 48)
        # shape: liste de (dx, dy) pré-calculés
        self.shape = shape

    def update(self):
        self.vy += 0.5
        self.vx *= 0.98
        self.cx += self.vx
        self.cy += self.vy
        self.angle += self.angular_vel
        self.angular_vel *= 0.98
        self.age += 1
        fade_start = self.max_age * 0.55
        if self.age > fade_start:
            d = self.max_age - fade_start
            if d > 0:
                self.opacity = max(0.0, 1.0 - (self.age - fade_start) / d)


class GlassShatterEffect:
    """Effet bris de glace performant."""

    def __init__(self):
        self.shards = []
        self._active = False
        self._flash_frames = 0
        self._flash_rect = None

    def trigger(self, frame, x1, y1, x2, y2, bw_active=False):
        h, w = frame.shape[:2]
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            return

        self.shards = []
        self._flash_frames = 4
        self._flash_rect = (x1, y1, x2, y2)

        cx_box = (x1 + x2) * 0.5
        cy_box = (y1 + y2) * 0.5
        diag = math.sqrt(bw * bw + bh * bh) * 0.5 + 1.0

        # Capturer la ROI pour échantillonner les couleurs
        roi = frame[y1:y2, x1:x2]
        if bw_active:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # ~20-30 fragments seulement
        num = min(30, max(12, (bw * bh) // 2500))
        for _ in range(num):
            # Position aléatoire dans le cadre
            sx = random.uniform(x1, x2)
            sy = random.uniform(y1, y2)

            # Échantillonner la couleur
            lx = int(np.clip(sx - x1, 0, bw - 1))
            ly = int(np.clip(sy - y1, 0, bh - 1))
            px = roi[ly, lx]
            # Rendre un peu plus lumineux (reflet de verre)
            boost = random.randint(15, 45)
            color = (min(255, int(px[0]) + boost),
                     min(255, int(px[1]) + boost),
                     min(255, int(px[2]) + boost))

            # Taille du fragment
            size = random.uniform(8, max(12, min(bw, bh) * 0.12))

            # Forme pré-calculée (triangle ou quad)
            nv = random.choice([3, 3, 4])
            shape = []
            base_a = random.uniform(0, 6.28)
            for i in range(nv):
                a = base_a + (6.2832 * i / nv) + random.uniform(-0.35, 0.35)
                r = size * random.uniform(0.5, 1.0)
                shape.append((r * math.cos(a), r * math.sin(a)))

            # Vélocité : explosion radiale depuis le centre
            dx = sx - cx_box
            dy = sy - cy_box
            dist = math.sqrt(dx * dx + dy * dy) + 0.01
            force = random.uniform(2.0, 6.0) * (0.4 + 0.6 * dist / diag)
            vx = (dx / dist) * force + random.uniform(-1, 1)
            vy = (dy / dist) * force * 0.5 + random.uniform(-3, -0.5)

            self.shards.append(GlassShard(sx, sy, vx, vy, size, color, shape))

        self._active = True

    @property
    def is_animating(self):
        return self._active

    def render(self, frame):
        if not self._active:
            return frame

        h, w = frame.shape[:2]

        # Flash blanc rapide (dessin direct, pas de copy)
        if self._flash_frames > 0 and self._flash_rect:
            fx1, fy1, fx2, fy2 = self._flash_rect
            intensity = self._flash_frames / 4.0
            # Éclaircir la zone du flash directement
            roi = frame[fy1:fy2, fx1:fx2]
            white = np.full_like(roi, 255, dtype=np.uint8)
            alpha = intensity * 0.45
            cv2.addWeighted(white, alpha, roi, 1.0 - alpha, 0, roi)
            frame[fy1:fy2, fx1:fx2] = roi
            self._flash_frames -= 1

        # Dessiner les fragments directement sur le frame
        alive = []
        cos_cache = {}
        sin_cache = {}

        for s in self.shards:
            s.update()
            if s.age >= s.max_age or s.opacity < 0.02:
                continue
            # Limites d'écran
            if s.cx < -50 or s.cx > w + 50 or s.cy > h + 50:
                continue

            alive.append(s)

            # Rotation des sommets
            ca = math.cos(s.angle)
            sa = math.sin(s.angle)
            pts = []
            for dx, dy in s.shape:
                rx = int(dx * ca - dy * sa + s.cx)
                ry = int(dx * sa + dy * ca + s.cy)
                pts.append((rx, ry))

            pts_arr = np.array(pts, dtype=np.int32)
            op = s.opacity

            # Couleur avec opacité simulée (assombrissement)
            c = (int(s.color[0] * op), int(s.color[1] * op), int(s.color[2] * op))

            # Remplissage du fragment
            cv2.fillConvexPoly(frame, pts_arr, c, cv2.LINE_AA)

            # Arête blanche fine (effet verre)
            edge = int(180 * op)
            cv2.polylines(frame, [pts_arr], True, (edge, edge, edge), 1, cv2.LINE_AA)

        self.shards = alive
        if len(self.shards) == 0 and self._flash_frames <= 0:
            self._active = False

        return frame
