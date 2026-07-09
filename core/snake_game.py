"""
CatchMyHands - Snake Game
=========================
Jeu de serpent contrôlé par la main.
Le serpent suit le bout de l'index à vitesse constante.
Inclus : collisions, croissance, particules, high score et écrans d'état.
"""

import cv2
import numpy as np
import math
import random
import time

class SnakeGame:
    """
    Gère la logique et le rendu du jeu Snake.
    """

    def __init__(self):
        self.high_score = 0
        self.score = 0
        self.segments = []  # Liste de tuples (x, y)
        self.game_over = False
        self.paused = False
        
        # Effets visuels
        self.particles = []  # Liste de dicts {x, y, vx, vy, life, max_life, color}
        self.floaters = []   # Liste de dicts {x, y, text, life, max_life, color}
        
        # Nourriture
        self.food = (0, 0)
        self.food_pulse = 0.0
        
        # Dimensions de jeu (initialisées au premier frame)
        self.width = 1280
        self.height = 720
        
        # Marges de l'arène de jeu
        self.margin_x = 60
        self.margin_y = 80
        
        self.initialized = False
    def reset(self, w: int, h: int):
        """Réinitialise une partie."""
        self.width = w
        self.height = h
        self.score = 0
        self.game_over = False
        self.paused = False
        self.particles = []
        self.floaters = []
        
        # Position de départ au centre
        start_x = w // 2
        start_y = h // 2
        
        # Créer le serpent initial (8 segments horizontaux vers la gauche)
        spacing = 15
        self.segments = []
        for i in range(8):
            self.segments.append((float(start_x - i * spacing), float(start_y)))
            
        self.angle = 0.0
        self.food = self._spawn_food()
        self.initialized = True

    def _spawn_food(self) -> tuple:
        """Génère une pomme dans une zone valide (loin du serpent)."""
        x_min = self.margin_x + 40
        x_max = self.width - self.margin_x - 40
        y_min = self.margin_y + 40
        y_max = self.height - self.margin_x - 40  # Marge égale en bas

        # Boucler jusqu'à trouver une position correcte
        for _ in range(100):
            fx = random.randint(x_min, x_max)
            fy = random.randint(y_min, y_max)
            
            # Vérifier que ce n'est pas sur le serpent
            too_close = False
            for seg in self.segments:
                dist = math.sqrt((seg[0] - fx) ** 2 + (seg[1] - fy) ** 2)
                if dist < 40:
                    too_close = True
                    break
            
            if not too_close:
                return (float(fx), float(fy))
                
        # Repli par défaut
        return (float(self.width // 2), float(self.height // 2 + 100))

    def _add_particle_burst(self, x: float, y: float, color: tuple):
        """Crée une explosion de particules neon."""
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            life = random.randint(15, 30)
            self.particles.append({
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "life": life,
                "max_life": life,
                "color": color
            })

    def _add_floating_text(self, x: float, y: float, text: str, color: tuple):
        """Crée un indicateur de score flottant."""
        self.floaters.append({
            "x": x,
            "y": y,
            "text": text,
            "life": 25,
            "max_life": 25,
            "color": color
        })

    def update_and_render(self, frame: np.ndarray, hands_metadata: list) -> np.ndarray:
        """
        Met à jour la physique du jeu et dessine l'écran complet.
        """
        h, w = frame.shape[:2]
        if not self.initialized:
            self.reset(w, h)

        # ── 1. Arrière-plan Cyberpunk de l'arène ──
        overlay = frame.copy()
        # Teinter légèrement l'ensemble
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 5, 10), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Dessiner le fond de l'arène de jeu (plus sombre)
        arena_x1 = self.margin_x
        arena_y1 = self.margin_y
        arena_x2 = w - self.margin_x
        arena_y2 = h - self.margin_x

        arena_bg = frame.copy()
        cv2.rectangle(arena_bg, (arena_x1, arena_y1), (arena_x2, arena_y2), (15, 10, 20), -1)
        cv2.addWeighted(arena_bg, 0.5, frame, 0.5, 0, frame)
        
        # Bordure de l'arène (Neon Violet)
        cv2.rectangle(frame, (arena_x1, arena_y1), (arena_x2, arena_y2), (255, 50, 255), 2, lineType=cv2.LINE_AA)

        # ── 2. Recherche de la main de contrôle ──
        active_hand_pos = None
        for hand_data in hands_metadata:
            # On prend la première main valide détectée
            smoothed = hand_data["smoothed"]
            idx_x = smoothed[8][0] * w
            idx_y = smoothed[8][1] * h
            active_hand_pos = (idx_x, idx_y)
            break

        # Gestion des états Pause / Game Over
        if self.game_over:
            active_hand_pos = None  # Désactiver le curseur
        elif active_hand_pos is None:
            self.paused = True
        else:
            self.paused = False

        # ── 3. Mise à jour de la physique (si actif) ──
        if not self.game_over and not self.paused and active_hand_pos is not None:
            target_x, target_y = active_hand_pos
            head_x, head_y = self.segments[0]

            dx = target_x - head_x
            dy = target_y - head_y
            dist = math.sqrt(dx ** 2 + dy ** 2)

            # Le serpent avance constamment
            # La vitesse augmente légèrement avec le score
            speed = 6.0 + min(6.0, self.score // 60.0)

            # Si le doigt est suffisamment éloigné, on met à jour la direction.
            # Sinon, le serpent continue tout droit dans sa direction actuelle
            # (ce qui l'empêche de s'arrêter lorsqu'il atteint le doigt).
            if dist > 20.0:
                self.angle = math.atan2(dy, dx)

            new_head_x = head_x + speed * math.cos(self.angle)
            new_head_y = head_y + speed * math.sin(self.angle)

            # Mise à jour des coordonnées avec l'algorithme de chaîne IK
            self.segments[0] = (new_head_x, new_head_y)
            spacing = 14.0
            for i in range(1, len(self.segments)):
                prev = self.segments[i - 1]
                curr = self.segments[i]
                cdx = curr[0] - prev[0]
                cdy = curr[1] - prev[1]
                cdist = math.sqrt(cdx ** 2 + cdy ** 2)
                if cdist > spacing:
                    ratio = spacing / cdist
                    self.segments[i] = (prev[0] + cdx * ratio, prev[1] + cdy * ratio)

            # ── Détection de collisions ──
            # 1. Collision avec les bords
            hx, hy = self.segments[0]
            if not (arena_x1 <= hx <= arena_x2 and arena_y1 <= hy <= arena_y2):
                self.game_over = True
                self._add_particle_burst(hx, hy, (50, 50, 255))
                self._add_floating_text(hx, hy - 20, "CRASH !", (0, 0, 255))

            # 2. Collision avec soi-même (exclure les 6 premiers segments pour la maniabilité)
            if not self.game_over:
                for i in range(6, len(self.segments)):
                    seg = self.segments[i]
                    s_dist = math.sqrt((hx - seg[0]) ** 2 + (hy - seg[1]) ** 2)
                    if s_dist < 12.0:
                        self.game_over = True
                        self._add_particle_burst(hx, hy, (50, 50, 255))
                        self._add_floating_text(hx, hy - 20, "AUTO-COLLISION !", (0, 0, 255))
                        break

            # 3. Collision avec la nourriture (manger)
            if not self.game_over:
                food_x, food_y = self.food
                f_dist = math.sqrt((hx - food_x) ** 2 + (hy - food_y) ** 2)
                if f_dist < 24.0:
                    self.score += 10
                    self.high_score = max(self.high_score, self.score)
                    self._add_particle_burst(food_x, food_y, (0, 255, 128))
                    self._add_floating_text(food_x, food_y - 20, "+10", (0, 255, 128))
                    
                    # Grandir (ajouter des segments)
                    last_seg = self.segments[-1]
                    for _ in range(3):
                        self.segments.append(last_seg)
                        
                    # Nouvelle nourriture
                    self.food = self._spawn_food()

        # ── 4. Mise à jour des effets secondaires (particules, floaters) ──
        # Particules
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vx"] *= 0.95  # friction
            p["vy"] *= 0.95
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

        # Textes flottants
        for f in self.floaters:
            f["y"] -= 1.0
            f["life"] -= 1
        self.floaters = [f for f in self.floaters if f["life"] > 0]

        # Pulsation de la nourriture
        self.food_pulse += 0.15
        if self.food_pulse > 2 * math.pi:
            self.food_pulse -= 2 * math.pi

        # ── 5. Rendu Visuel ──
        # Dessiner la nourriture
        food_x, food_y = self.food
        pulse_scale = 1.0 + 0.2 * math.sin(self.food_pulse)
        food_color = (0, 200, 255)  # Jaune/Orange néon
        food_glow_color = (0, 100, 150)
        # Lueur externe
        cv2.circle(frame, (int(food_x), int(food_y)), int(22 * pulse_scale), food_glow_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (int(food_x), int(food_y)), int(8 * pulse_scale), (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (int(food_x), int(food_y)), int(6 * pulse_scale), food_color, 2, lineType=cv2.LINE_AA)

        # Dessiner le serpent (du bout de la queue à la tête pour un bon empilage)
        num_segs = len(self.segments)
        for i in reversed(range(num_segs)):
            seg = self.segments[i]
            x, y = int(seg[0]), int(seg[1])
            
            # Dégradé de couleur et rayon décroissant
            # De la tête verte (0, 255, 128) à la queue cyan (255, 255, 0)
            t = i / max(1, num_segs - 1)
            radius = int(12 - 5 * t)
            
            color_r = int((0 * (1 - t)) + (255 * t))
            color_g = int((255 * (1 - t)) + (255 * t))
            color_b = int((128 * (1 - t)) + (0 * t))
            color = (color_b, color_g, color_r)
            
            if i == 0:
                # Tête (un peu plus grosse)
                # Lueur
                cv2.circle(frame, (x, y), radius + 6, (0, 150, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, (x, y), radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, (x, y), radius - 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                
                # Dessiner les yeux en direction de la cible
                if active_hand_pos is not None:
                    tx, ty = active_hand_pos
                else:
                    tx, ty = x + 10, y  # Regard par défaut vers la droite
                
                eye_dx = tx - x
                eye_dy = ty - y
                eye_angle = math.atan2(eye_dy, eye_dx)
                
                eye_offset = 6
                eye_spread = 0.5
                eye_radius = 4
                pupil_radius = 2
                
                # Oeil gauche
                el_x = int(x + eye_offset * math.cos(eye_angle - eye_spread))
                el_y = int(y + eye_offset * math.sin(eye_angle - eye_spread))
                cv2.circle(frame, (el_x, el_y), eye_radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                # Pupil
                pl_x = int(el_x + 1 * math.cos(eye_angle))
                pl_y = int(el_y + 1 * math.sin(eye_angle))
                cv2.circle(frame, (pl_x, pl_y), pupil_radius, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                
                # Oeil droit
                er_x = int(x + eye_offset * math.cos(eye_angle + eye_spread))
                er_y = int(y + eye_offset * math.sin(eye_angle + eye_spread))
                cv2.circle(frame, (er_x, er_y), eye_radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                # Pupil
                pr_x = int(er_x + 1 * math.cos(eye_angle))
                pr_y = int(er_y + 1 * math.sin(eye_angle))
                cv2.circle(frame, (pr_x, pr_y), pupil_radius, (0, 0, 0), -1, lineType=cv2.LINE_AA)
            else:
                # Corps
                # Petit glow
                cv2.circle(frame, (x, y), radius + 3, (int(color[0]*0.4), int(color[1]*0.4), int(color[2]*0.4)), -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

        # Dessiner le curseur cible de la main
        if active_hand_pos is not None and not self.game_over:
            cx, cy = int(active_hand_pos[0]), int(active_hand_pos[1])
            # Petite cible holographique verte
            cv2.circle(frame, (cx, cy), 16, (0, 255, 128), 1, lineType=cv2.LINE_AA)
            cv2.line(frame, (cx - 22, cy), (cx - 8, cy), (0, 255, 128), 1, lineType=cv2.LINE_AA)
            cv2.line(frame, (cx + 8, cy), (cx + 22, cy), (0, 255, 128), 1, lineType=cv2.LINE_AA)
            cv2.line(frame, (cx, cy - 22), (cx, cy - 8), (0, 255, 128), 1, lineType=cv2.LINE_AA)
            cv2.line(frame, (cx, cy + 8), (cx, cy + 22), (0, 255, 128), 1, lineType=cv2.LINE_AA)

        # Dessiner les particules
        for p in self.particles:
            alpha = p["life"] / p["max_life"]
            px, py = int(p["x"]), int(p["y"])
            p_radius = max(1, int(4 * alpha))
            cv2.circle(frame, (px, py), p_radius, p["color"], -1, lineType=cv2.LINE_AA)

        # Dessiner les textes flottants
        font = cv2.FONT_HERSHEY_SIMPLEX
        for f in self.floaters:
            alpha = f["life"] / f["max_life"]
            fx, fy = int(f["x"]), int(f["y"])
            cv2.putText(frame, f["text"], (fx, fy), font, 0.5, f["color"], 2, cv2.LINE_AA)

        # Scoreboard du HUD
        hud_text = f"SCORE : {self.score:04d}   MEILLEUR : {self.high_score:04d}"
        cv2.putText(frame, hud_text, (self.margin_x + 10, self.margin_y - 15), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        
        controls_desc = "[ M / ESC : Menu Principal ]  [ R : Recommencer ]"
        cv2.putText(frame, controls_desc, (w - self.margin_x - 380, self.margin_y - 15), font, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

        # ── 6. Overlays d'état (Pause / Game Over) ──
        if self.paused and not self.game_over:
            p_overlay = frame.copy()
            cv2.rectangle(p_overlay, (arena_x1, arena_y1), (arena_x2, arena_y2), (30, 20, 10), -1)
            cv2.addWeighted(p_overlay, 0.55, frame, 0.45, 0, frame)
            
            # Alerte
            cv2.rectangle(frame, (w // 2 - 250, h // 2 - 60), (w // 2 + 250, h // 2 + 60), (0, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, "JEU EN PAUSE", (w // 2 - 110, h // 2 - 10), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Placez votre main devant la camera", (w // 2 - 210, h // 2 + 25), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "L'index sert a guider le serpent", (w // 2 - 170, h // 2 + 45), font, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        if self.game_over:
            go_overlay = frame.copy()
            cv2.rectangle(go_overlay, (arena_x1, arena_y1), (arena_x2, arena_y2), (10, 10, 40), -1)
            cv2.addWeighted(go_overlay, 0.7, frame, 0.3, 0, frame)

            # Boite Game Over
            cv2.rectangle(frame, (w // 2 - 260, h // 2 - 90), (w // 2 + 260, h // 2 + 90), (50, 50, 255), 2, lineType=cv2.LINE_AA)
            
            # Lueur rouge
            cv2.putText(frame, "GAME OVER", (w // 2 - 130, h // 2 - 30), font, 1.2, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, "GAME OVER", (w // 2 - 130, h // 2 - 30), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

            score_text = f"Score final : {self.score}"
            cv2.putText(frame, score_text, (w // 2 - 75, h // 2 + 10), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.putText(frame, "[ R ]  Recommencer une partie", (w // 2 - 170, h // 2 + 45), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "[ M ]  Retourner au Menu Principal", (w // 2 - 180, h // 2 + 68), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        return frame
