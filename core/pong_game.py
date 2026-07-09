"""
CatchMyHands - Pong Game
========================
Jeu de Pong contrôlé par la main avec un adversaire IA.
Le joueur contrôle la raquette du bas avec son index.
L'IA contrôle la raquette du haut avec un tracking imparfait.
Inclus : power-ups, multi-ball, particules, vies, score.
"""

import cv2
import numpy as np
import math
import random
import time


class PongGame:
    """
    Gère la logique et le rendu du jeu Pong avec IA et power-ups.
    """

    # Types de power-ups
    POWERUP_EXTEND = "extend"
    POWERUP_SLOWMO = "slowmo"
    POWERUP_SHIELD = "shield"
    POWERUP_SPEED = "speed"
    POWERUP_MULTI = "multi"

    POWERUP_INFO = {
        "extend":  {"label": "EXTEND",     "color": (0, 255, 128),   "icon": "+", "duration_frames": 480},
        "slowmo":  {"label": "SLOW-MO",    "color": (255, 200, 0),   "icon": "~", "duration_frames": 360},
        "shield":  {"label": "SHIELD",     "color": (255, 255, 0),   "icon": "=", "duration_frames": 0},
        "speed":   {"label": "SPEED !",    "color": (0, 80, 255),    "icon": "!", "duration_frames": 300},
        "multi":   {"label": "MULTI-BALL", "color": (255, 50, 255),  "icon": "*", "duration_frames": 0},
    }

    def __init__(self):
        self.high_score = 0
        self.initialized = False

        # Charger l'image du coeur pixelisé
        self.heart_full = None
        self.heart_empty = None
        self.heart_size = 20
        self.heart_mask = None

        try:
            import os
            # Charger depuis le dossier 'img' ou le dossier courant
            img_path = "img/pixel-heart.jpg"
            if not os.path.exists(img_path):
                # Essayer avec le chemin complet si nécessaire
                img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "img", "pixel-heart.jpg")

            heart_img = cv2.imread(img_path)
            if heart_img is not None:
                # Convertir en niveaux de gris pour trouver le contour du coeur
                gray = cv2.cvtColor(heart_img, cv2.COLOR_BGR2GRAY)
                # Le fond est blanc (proche de 255), on seuille pour isoler le coeur sombre
                _, thresh = cv2.threshold(gray, 242, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Prendre le plus grand contour (le coeur)
                    c = max(contours, key=cv2.contourArea)
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    # Rogner (crop) les bords blancs inutiles
                    heart_img = heart_img[y:y+h_box, x:x+w_box]

                # Redimensionner en 20x20
                self.heart_full = cv2.resize(heart_img, (self.heart_size, self.heart_size), interpolation=cv2.INTER_AREA)

                # Masque de transparence (pixels non blancs)
                gray_heart = cv2.cvtColor(self.heart_full, cv2.COLOR_BGR2GRAY)
                _, self.heart_mask = cv2.threshold(gray_heart, 242, 255, cv2.THRESH_BINARY_INV)

                # Créer le coeur vide grisé et sombre (couleur gris cyberpunk)
                grey = cv2.cvtColor(self.heart_full, cv2.COLOR_BGR2GRAY)
                grey_3ch = cv2.merge([grey, grey, grey])
                self.heart_empty = (grey_3ch * 0.35).astype(np.uint8)
        except Exception as e:
            print(f"[PongGame] Erreur lors du chargement ou du crop du coeur : {e}")

        # Fallback si l'image n'a pas pu être chargée ou traitée
        if self.heart_full is None:
            self.heart_full = np.zeros((self.heart_size, self.heart_size, 3), dtype=np.uint8)
            self.heart_full[:, :] = (0, 0, 255) # Rouge
            self.heart_mask = np.ones((self.heart_size, self.heart_size), dtype=np.uint8) * 255
            self.heart_empty = np.zeros((self.heart_size, self.heart_size, 3), dtype=np.uint8)
            self.heart_empty[:, :] = (60, 60, 60) # Gris

    def reset(self, w: int, h: int):
        """Réinitialise une partie de Pong."""
        self.width = w
        self.height = h

        # Marges de l'arène
        self.margin_x = 60
        self.margin_y = 60
        self.arena_x1 = self.margin_x
        self.arena_y1 = self.margin_y
        self.arena_x2 = w - self.margin_x
        self.arena_y2 = h - self.margin_y
        self.arena_w = self.arena_x2 - self.arena_x1
        self.arena_h = self.arena_y2 - self.arena_y1

        # État du jeu
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.paused = False

        # ── Raquette joueur (bas) ──
        self.paddle_base_width = 130
        self.paddle_width = self.paddle_base_width
        self.paddle_height = 14
        self.paddle_x = float(w // 2)
        self.paddle_y = float(self.arena_y2 - 35)

        # ── Raquette IA (haut) ──
        self.ai_paddle_width = 120
        self.ai_paddle_height = 14
        self.ai_paddle_x = float(w // 2)
        self.ai_paddle_y = float(self.arena_y1 + 35)
        self.ai_speed = 4.5           # Vitesse de base de l'IA
        self.ai_error_offset = 0.0    # Erreur intentionnelle
        self.ai_error_timer = 0       # Timer pour changer l'erreur

        # ── Balles ──
        self.balls = []
        self._spawn_ball()

        # ── Power-ups ──
        self.active_powerup = None        # {type, x, y, spawn_frame}
        self.powerup_effects = {}         # {type: expire_frame}
        self.powerup_cooldown = 180       # Frames avant le premier spawn
        self.shield_active = False
        self.frame_counter = 0

        # ── Effets visuels ──
        self.particles = []
        self.floaters = []
        self.ball_trails = []             # [{x, y, life}]

        self.initialized = True

    def _spawn_ball(self):
        """Crée une balle au centre de l'arène avec une direction aléatoire."""
        cx = float(self.width // 2)
        cy = float(self.height // 2)
        # Angle aléatoire vers le bas (joueur) ou vers le haut (IA)
        angle = random.choice([
            random.uniform(math.radians(200), math.radians(340)),  # Vers le bas
            random.uniform(math.radians(20), math.radians(160)),   # Vers le haut
        ])
        speed = 5.0
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        # Éviter les trajectoires trop horizontales
        if abs(vy) < 2.0:
            vy = 2.5 if vy >= 0 else -2.5
        self.balls.append({"x": cx, "y": cy, "vx": vx, "vy": vy})

    def _spawn_powerup(self):
        """Fait apparaître un power-up aléatoire dans l'arène."""
        ptype = random.choice([
            self.POWERUP_EXTEND, self.POWERUP_SLOWMO,
            self.POWERUP_SHIELD, self.POWERUP_SPEED, self.POWERUP_MULTI
        ])
        margin = 80
        px = random.randint(self.arena_x1 + margin, self.arena_x2 - margin)
        py = random.randint(self.arena_y1 + 100, self.arena_y2 - 100)
        self.active_powerup = {
            "type": ptype, "x": float(px), "y": float(py),
            "spawn_frame": self.frame_counter
        }

    def _activate_powerup(self, ptype: str, ball: dict):
        """Active l'effet d'un power-up."""
        info = self.POWERUP_INFO[ptype]
        px = self.active_powerup["x"]
        py = self.active_powerup["y"]

        self._add_particle_burst(px, py, info["color"], count=20)
        self._add_floating_text(px, py - 20, info["label"], info["color"])

        if ptype == self.POWERUP_EXTEND:
            self.paddle_width = int(self.paddle_base_width * 2.0)
            self.powerup_effects[ptype] = self.frame_counter + info["duration_frames"]

        elif ptype == self.POWERUP_SLOWMO:
            # Ralentir toutes les balles
            for b in self.balls:
                b["vx"] *= 0.5
                b["vy"] *= 0.5
            self.powerup_effects[ptype] = self.frame_counter + info["duration_frames"]

        elif ptype == self.POWERUP_SHIELD:
            self.shield_active = True

        elif ptype == self.POWERUP_SPEED:
            for b in self.balls:
                b["vx"] *= 1.8
                b["vy"] *= 1.8
            self.powerup_effects[ptype] = self.frame_counter + info["duration_frames"]

        elif ptype == self.POWERUP_MULTI:
            self._spawn_ball()

        self.active_powerup = None

    def _expire_powerups(self):
        """Vérifie et expire les power-ups temporaires."""
        expired = []
        for ptype, expire_frame in self.powerup_effects.items():
            if self.frame_counter >= expire_frame:
                expired.append(ptype)

        for ptype in expired:
            del self.powerup_effects[ptype]
            if ptype == self.POWERUP_EXTEND:
                self.paddle_width = self.paddle_base_width
            elif ptype == self.POWERUP_SLOWMO:
                # Restaurer la vitesse (accélérer de retour)
                for b in self.balls:
                    b["vx"] *= 2.0
                    b["vy"] *= 2.0
            elif ptype == self.POWERUP_SPEED:
                for b in self.balls:
                    b["vx"] /= 1.8
                    b["vy"] /= 1.8

    def _add_particle_burst(self, x, y, color, count=12):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            life = random.randint(15, 30)
            self.particles.append({
                "x": x, "y": y,
                "vx": speed * math.cos(angle), "vy": speed * math.sin(angle),
                "life": life, "max_life": life, "color": color
            })

    def _add_floating_text(self, x, y, text, color):
        self.floaters.append({
            "x": x, "y": y, "text": text,
            "life": 30, "max_life": 30, "color": color
        })

    # ──────────────────────────────────────────────────────
    # MAIN UPDATE & RENDER
    # ──────────────────────────────────────────────────────
    def update_and_render(self, frame: np.ndarray, hands_metadata: list) -> np.ndarray:
        """Met à jour la physique et dessine le jeu complet."""
        h, w = frame.shape[:2]
        if not self.initialized:
            self.reset(w, h)

        self.frame_counter += 1

        # ── 1. Fond cyberpunk de l'arène ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 5, 15), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        arena_bg = frame.copy()
        cv2.rectangle(arena_bg, (self.arena_x1, self.arena_y1),
                      (self.arena_x2, self.arena_y2), (15, 10, 25), -1)
        cv2.addWeighted(arena_bg, 0.5, frame, 0.5, 0, frame)

        # Bordure arène
        cv2.rectangle(frame, (self.arena_x1, self.arena_y1),
                      (self.arena_x2, self.arena_y2), (255, 100, 0), 2, cv2.LINE_AA)

        # Ligne médiane pointillée
        mid_y = (self.arena_y1 + self.arena_y2) // 2
        dash_len = 15
        for x in range(self.arena_x1 + 20, self.arena_x2 - 20, dash_len * 2):
            cv2.line(frame, (x, mid_y), (x + dash_len, mid_y), (60, 40, 30), 1, cv2.LINE_AA)

        # ── 2. Position de la main ──
        hand_x = None
        for hand_data in hands_metadata:
            smoothed = hand_data["smoothed"]
            hand_x = smoothed[8][0] * w  # Index finger tip X
            break

        if self.game_over:
            hand_x = None
        elif hand_x is None:
            self.paused = True
        else:
            self.paused = False

        # ── 3. Mise à jour de la physique ──
        if not self.game_over and not self.paused and hand_x is not None:
            # Déplacer la raquette joueur (lissage)
            target_x = max(self.arena_x1 + self.paddle_width // 2,
                           min(self.arena_x2 - self.paddle_width // 2, hand_x))
            self.paddle_x += (target_x - self.paddle_x) * 0.35

            # Mettre à jour l'IA
            self._update_ai()

            # Expirer les power-ups
            self._expire_powerups()

            # Spawn de power-ups
            if self.active_powerup is None:
                self.powerup_cooldown -= 1
                if self.powerup_cooldown <= 0:
                    self._spawn_powerup()
                    self.powerup_cooldown = random.randint(300, 600)
            else:
                # Expirer le power-up visible après 10s (600 frames)
                if self.frame_counter - self.active_powerup["spawn_frame"] > 600:
                    self.active_powerup = None
                    self.powerup_cooldown = random.randint(120, 300)

            # Mettre à jour chaque balle
            balls_to_remove = []
            for bi, ball in enumerate(self.balls):
                result = self._update_ball(ball)
                if result == "lost":
                    balls_to_remove.append(bi)
                elif result == "scored":
                    # Balle passée derrière l'IA
                    self.score += 1
                    self.high_score = max(self.high_score, self.score)
                    self._add_floating_text(ball["x"], self.arena_y1 + 15, "+1", (0, 255, 128))
                    self._add_particle_burst(ball["x"], self.arena_y1 + 10, (0, 255, 128))
                    balls_to_remove.append(bi)

            # Retirer les balles perdues/marquées (en ordre inverse)
            for bi in sorted(balls_to_remove, reverse=True):
                self.balls.pop(bi)

            # Si plus aucune balle
            if len(self.balls) == 0:
                if self.lives > 0:
                    self._spawn_ball()
                else:
                    self.game_over = True

        # ── 4. Effets secondaires ──
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vx"] *= 0.94
            p["vy"] *= 0.94
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

        for f in self.floaters:
            f["y"] -= 1.2
            f["life"] -= 1
        self.floaters = [f for f in self.floaters if f["life"] > 0]

        # Trails
        for t in self.ball_trails:
            t["life"] -= 1
        self.ball_trails = [t for t in self.ball_trails if t["life"] > 0]

        # ── 5. Rendu visuel ──
        self._draw_shield(frame)
        self._draw_paddle(frame, self.paddle_x, self.paddle_y,
                          self.paddle_width, self.paddle_height, (0, 200, 255), "PLAYER")
        self._draw_paddle(frame, self.ai_paddle_x, self.ai_paddle_y,
                          self.ai_paddle_width, self.ai_paddle_height, (0, 100, 255), "IA")
        self._draw_powerup(frame)
        self._draw_trails(frame)
        self._draw_balls(frame)
        self._draw_particles(frame)
        self._draw_floaters(frame)
        self._draw_hud(frame)

        # Curseur main
        if hand_x is not None and not self.game_over:
            cx = int(hand_x)
            cy = int(self.paddle_y)
            cv2.line(frame, (cx, self.arena_y2 - 8), (cx, self.arena_y2 + 2),
                     (0, 255, 128), 2, cv2.LINE_AA)

        # ── 6. Overlays d'état ──
        if self.paused and not self.game_over:
            self._draw_pause_overlay(frame)
        if self.game_over:
            self._draw_gameover_overlay(frame)

        return frame

    def _update_ai(self):
        """Met à jour la position de la raquette IA."""
        if not self.balls:
            return

        # Trouver la balle la plus proche du haut (menace)
        threat_ball = None
        min_y = float('inf')
        for b in self.balls:
            if b["vy"] < 0 and b["y"] < min_y:  # Balle montante
                min_y = b["y"]
                threat_ball = b

        if threat_ball is None:
            # Pas de menace directe, suivre la première balle doucement
            threat_ball = self.balls[0]

        # Erreur intentionnelle (change toutes les 2-3 secondes)
        self.ai_error_timer -= 1
        if self.ai_error_timer <= 0:
            self.ai_error_offset = random.uniform(-40, 40)
            self.ai_error_timer = random.randint(120, 180)

        target_x = threat_ball["x"] + self.ai_error_offset

        # Vitesse de l'IA augmente avec le score du joueur
        current_ai_speed = self.ai_speed + min(3.0, self.score * 0.15)

        # Déplacement avec limitation de vitesse
        diff = target_x - self.ai_paddle_x
        move = max(-current_ai_speed, min(current_ai_speed, diff))
        self.ai_paddle_x += move

        # Clamper dans l'arène
        self.ai_paddle_x = max(self.arena_x1 + self.ai_paddle_width // 2,
                               min(self.arena_x2 - self.ai_paddle_width // 2, self.ai_paddle_x))

    def _update_ball(self, ball: dict) -> str:
        """Met à jour une balle. Retourne 'lost', 'scored', ou 'ok'."""
        # Sauvegarder la position Y précédente pour l'anti-tunneling
        prev_y = ball["y"]

        ball["x"] += ball["vx"]
        ball["y"] += ball["vy"]

        # Trail
        self.ball_trails.append({"x": ball["x"], "y": ball["y"], "life": 8})

        bx, by = ball["x"], ball["y"]
        radius = 8

        # ── Rebond murs gauche/droite ──
        if bx - radius <= self.arena_x1:
            ball["x"] = self.arena_x1 + radius
            ball["vx"] = abs(ball["vx"])
            self._add_particle_burst(ball["x"], ball["y"], (255, 100, 0), 6)
        elif bx + radius >= self.arena_x2:
            ball["x"] = self.arena_x2 - radius
            ball["vx"] = -abs(ball["vx"])
            self._add_particle_burst(ball["x"], ball["y"], (255, 100, 0), 6)

        # ── Collision raquette joueur (bas) ──
        p_left = self.paddle_x - self.paddle_width / 2
        p_right = self.paddle_x + self.paddle_width / 2
        p_top = self.paddle_y - self.paddle_height / 2

        # Vérification anti-tunneling par franchissement de ligne
        if (ball["vy"] > 0 and
                prev_y + radius <= p_top + 3 and
                by + radius >= p_top and
                p_left - 5 <= bx <= p_right + 5):
            ball["y"] = p_top - radius
            # Angle de rebond basé sur où la balle touche la raquette
            hit_pos = (bx - self.paddle_x) / (self.paddle_width / 2)
            hit_pos = max(-1.0, min(1.0, hit_pos))
            speed = math.sqrt(ball["vx"] ** 2 + ball["vy"] ** 2)
            # Accélérer de +0.5 à chaque rebond
            speed = min(22.0, speed + 0.5)
            angle = math.radians(-90 + hit_pos * 55)
            ball["vx"] = speed * math.cos(angle)
            ball["vy"] = speed * math.sin(angle)
            self._add_particle_burst(bx, p_top, (0, 200, 255), 8)

        # ── Collision raquette IA (haut) ──
        ai_left = self.ai_paddle_x - self.ai_paddle_width / 2
        ai_right = self.ai_paddle_x + self.ai_paddle_width / 2
        ai_bottom = self.ai_paddle_y + self.ai_paddle_height / 2

        # Vérification anti-tunneling par franchissement de ligne
        if (ball["vy"] < 0 and
                prev_y - radius >= ai_bottom - 3 and
                by - radius <= ai_bottom and
                ai_left - 5 <= bx <= ai_right + 5):
            ball["y"] = ai_bottom + radius
            hit_pos = (bx - self.ai_paddle_x) / (self.ai_paddle_width / 2)
            hit_pos = max(-1.0, min(1.0, hit_pos))
            speed = math.sqrt(ball["vx"] ** 2 + ball["vy"] ** 2)
            # Accélérer de +0.5 à chaque rebond
            speed = min(22.0, speed + 0.5)
            angle = math.radians(90 + hit_pos * 55)
            ball["vx"] = speed * math.cos(angle)
            ball["vy"] = speed * math.sin(angle)
            self._add_particle_burst(bx, ai_bottom, (0, 100, 255), 8)

        # ── Balle passée en bas (perte de vie) ──
        if by > self.arena_y2 + 10:
            # Vérifier le shield
            if self.shield_active:
                ball["vy"] = -abs(ball["vy"])
                ball["y"] = self.arena_y2 - radius
                self.shield_active = False
                self._add_particle_burst(bx, self.arena_y2, (255, 255, 0), 15)
                self._add_floating_text(bx, self.arena_y2 - 30, "SHIELD!", (255, 255, 0))
                return "ok"
            else:
                self.lives -= 1
                self._add_particle_burst(bx, self.arena_y2, (0, 0, 255), 15)
                self._add_floating_text(bx, self.arena_y2 - 30, "-1 VIE", (0, 50, 255))
                return "lost"

        # ── Balle passée en haut (joueur marque) ──
        if by < self.arena_y1 - 10:
            return "scored"

        # ── Collision power-up ──
        if self.active_powerup is not None:
            px, py = self.active_powerup["x"], self.active_powerup["y"]
            dist = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
            if dist < 28:
                self._activate_powerup(self.active_powerup["type"], ball)

        return "ok"

    # ──────────────────────────────────────────────────────
    # DRAWING METHODS
    # ──────────────────────────────────────────────────────
    def _draw_paddle(self, frame, cx, cy, pw, ph, color, label):
        """Dessine une raquette avec effet néon."""
        x1 = int(cx - pw / 2)
        y1 = int(cy - ph / 2)
        x2 = int(cx + pw / 2)
        y2 = int(cy + ph / 2)

        # Lueur extérieure
        glow_color = (color[0] // 3, color[1] // 3, color[2] // 3)
        cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                      glow_color, -1, cv2.LINE_AA)
        # Corps
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
        # Bordure blanche
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_balls(self, frame):
        """Dessine toutes les balles actives."""
        for ball in self.balls:
            bx, by = int(ball["x"]), int(ball["y"])
            radius = 8
            # Lueur extérieure
            cv2.circle(frame, (bx, by), radius + 5, (0, 80, 120), -1, cv2.LINE_AA)
            # Corps
            cv2.circle(frame, (bx, by), radius, (0, 220, 255), -1, cv2.LINE_AA)
            # Reflet
            cv2.circle(frame, (bx - 2, by - 2), 3, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_trails(self, frame):
        """Dessine les traînées de la balle."""
        for t in self.ball_trails:
            alpha = t["life"] / 8.0
            r = max(1, int(5 * alpha))
            c = int(200 * alpha)
            cv2.circle(frame, (int(t["x"]), int(t["y"])), r,
                       (0, c // 2, c), -1, cv2.LINE_AA)

    def _draw_shield(self, frame):
        """Dessine le filet de sécurité shield si actif."""
        if self.shield_active:
            y = self.arena_y2 + 5
            # Ligne dorée scintillante
            pulse = 0.7 + 0.3 * math.sin(self.frame_counter * 0.2)
            color = (0, int(255 * pulse), int(255 * pulse))
            cv2.line(frame, (self.arena_x1 + 10, y), (self.arena_x2 - 10, y),
                     color, 3, cv2.LINE_AA)
            cv2.putText(frame, "SHIELD ACTIVE", (self.arena_x1 + 10, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    def _draw_powerup(self, frame):
        """Dessine le power-up visible sur le terrain."""
        if self.active_powerup is None:
            return

        px, py = int(self.active_powerup["x"]), int(self.active_powerup["y"])
        ptype = self.active_powerup["type"]
        info = self.POWERUP_INFO[ptype]
        color = info["color"]

        # Pulsation
        pulse = 1.0 + 0.15 * math.sin(self.frame_counter * 0.15)
        radius = int(18 * pulse)

        # Lueur
        glow = (color[0] // 4, color[1] // 4, color[2] // 4)
        cv2.circle(frame, (px, py), radius + 8, glow, -1, cv2.LINE_AA)
        # Cercle principal
        cv2.circle(frame, (px, py), radius, color, 2, cv2.LINE_AA)
        # Icône
        cv2.putText(frame, info["icon"], (px - 5, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        # Label en dessous
        cv2.putText(frame, info["label"], (px - 30, py + radius + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # Timer visuel (barre de progression restante)
        elapsed = self.frame_counter - self.active_powerup["spawn_frame"]
        remaining = max(0.0, 1.0 - elapsed / 600.0)
        bar_w = 40
        bar_x = px - bar_w // 2
        bar_y = py + radius + 25
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 4), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * remaining), bar_y + 4),
                      color, -1)

    def _draw_particles(self, frame):
        for p in self.particles:
            alpha = p["life"] / p["max_life"]
            r = max(1, int(4 * alpha))
            cv2.circle(frame, (int(p["x"]), int(p["y"])), r,
                       p["color"], -1, cv2.LINE_AA)

    def _draw_floaters(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for f in self.floaters:
            cv2.putText(frame, f["text"], (int(f["x"]), int(f["y"])),
                        font, 0.55, f["color"], 2, cv2.LINE_AA)

    def _draw_hud(self, frame):
        """Dessine le scoreboard et les vies."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Score
        score_text = f"SCORE : {self.score:03d}   MEILLEUR : {self.high_score:03d}"
        cv2.putText(frame, score_text, (self.margin_x + 10, self.margin_y - 15),
                    font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Vies (cœurs sous forme d'images pixelisées avec masque de transparence)
        cv2.putText(frame, "VIES: ", (self.arena_x2 - 160, self.margin_y - 15),
                    font, 0.5, (0, 128, 255) if self.lives > 1 else (0, 0, 255), 2, cv2.LINE_AA)
        
        # Position de départ des cœurs (juste après le texte "VIES:")
        start_x = self.arena_x2 - 100
        start_y = self.margin_y - 28
        spacing = 26
        
        for i in range(3):
            px = start_x + i * spacing
            py = start_y
            # Choisir l'image (coeur plein ou vide grisé)
            heart_img = self.heart_full if i < self.lives else self.heart_empty
            
            # S'assurer que les coordonnées restent dans les limites du frame
            if 0 <= px < self.width - self.heart_size and 0 <= py < self.height - self.heart_size:
                roi = frame[py:py+self.heart_size, px:px+self.heart_size]
                # Appliquer le masque pour la transparence
                roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(self.heart_mask))
                roi_fg = cv2.bitwise_and(heart_img, heart_img, mask=self.heart_mask)
                frame[py:py+self.heart_size, px:px+self.heart_size] = cv2.add(roi_bg, roi_fg)

        # Power-ups actifs (timers)
        y_offset = self.margin_y - 15
        x_offset = self.width // 2 - 60
        for ptype, expire in self.powerup_effects.items():
            remaining_s = max(0, (expire - self.frame_counter)) / 60.0
            info = self.POWERUP_INFO[ptype]
            timer_text = f"{info['label']} {remaining_s:.1f}s"
            cv2.putText(frame, timer_text, (x_offset, y_offset),
                        font, 0.4, info["color"], 1, cv2.LINE_AA)
            x_offset += 130

        # Contrôles
        controls = "[ M/ESC : Menu ]  [ R : Recommencer ]"
        cv2.putText(frame, controls, (self.width - self.margin_x - 340, self.height - self.margin_y + 25),
                    font, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

    def _draw_pause_overlay(self, frame):
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (self.arena_x1, self.arena_y1),
                      (self.arena_x2, self.arena_y2), (30, 20, 10), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (w // 2 - 250, h // 2 - 60),
                      (w // 2 + 250, h // 2 + 60), (255, 180, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "JEU EN PAUSE", (w // 2 - 110, h // 2 - 10),
                    font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Placez votre main devant la camera",
                    (w // 2 - 210, h // 2 + 25), font, 0.5, (255, 180, 0), 1, cv2.LINE_AA)

    def _draw_gameover_overlay(self, frame):
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (self.arena_x1, self.arena_y1),
                      (self.arena_x2, self.arena_y2), (10, 10, 40), -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (w // 2 - 260, h // 2 - 90),
                      (w // 2 + 260, h // 2 + 90), (50, 50, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "GAME OVER", (w // 2 - 130, h // 2 - 30),
                    font, 1.2, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, "GAME OVER", (w // 2 - 130, h // 2 - 30),
                    font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Score final : {self.score}",
                    (w // 2 - 75, h // 2 + 10), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "[ R ]  Recommencer une partie",
                    (w // 2 - 170, h // 2 + 45), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "[ M ]  Retourner au Menu Jeux",
                    (w // 2 - 170, h // 2 + 68), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
