"""
CatchMyHands - Games Selection Menu Renderer
=============================================
Affiche un sous-menu cyberpunk permettant de choisir entre les jeux disponibles
(Snake et Pong), par raccourci clavier ou en pointant/pinchant avec la main.
"""

import cv2
import numpy as np
import config
from core.gesture_engine import GestureType

class GamesMenuRenderer:
    """
    Rend le menu de sélection des jeux et gère l'interaction gestuelle.
    """

    def __init__(self):
        # Mémoriser les états de clic pour éviter les déclenchements répétitifs
        # Initialisé à True pour forcer l'utilisateur à ouvrir sa main avant de cliquer
        self.was_pinching = True

    def render(self, frame: np.ndarray, hands_metadata: list, block_clicks: bool = False) -> tuple:
        """
        Dessine le menu des jeux et traite les entrées gestuelles.

        Args:
            frame: Image BGR OpenCV de la caméra.
            hands_metadata: Liste des métadonnées des mains détectées ce frame.

        Returns:
            Un tuple (frame_modifié, action_choisie)
            action_choisie vaut :
                - None : aucune action
                - 1 : Lancer Snake
                - 2 : Lancer Pong
        """
        h, w = frame.shape[:2]
        
        # ── 1. Arrière-plan Cyberpunk (Sombre + Grille) ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (15, 5, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Dessiner des lignes de grille
        grid_size = 60
        for x in range(0, w, grid_size):
            cv2.line(frame, (x, 0), (x, h), (40, 20, 40), 1)
        for y in range(0, h, grid_size):
            cv2.line(frame, (0, y), (w, y), (40, 20, 40), 1)

        # ── 2. Définition des zones de boutons side-by-side ──
        box_w = int(w * 0.36)
        box_h = int(h * 0.42)
        y1 = int(h * 0.38)
        y2 = y1 + box_h

        # Bouton 1 : Snake (à gauche)
        btn1_x1 = int(w * 0.10)
        btn1_x2 = btn1_x1 + box_w

        # Bouton 2 : Pong (à droite)
        btn2_x1 = int(w * 0.54)
        btn2_x2 = btn2_x1 + box_w

        # ── 3. Analyse des positions de mains (Hover & Pinch) ──
        hover1 = False
        hover2 = False
        action_triggered = None
        is_currently_pinching = False

        # Si aucune main n'est détectée, on réinitialise l'état de clic en bloquant le prochain premier frame
        if not hands_metadata:
            self.was_pinching = True

        for hand_data in hands_metadata:
            smoothed = hand_data["smoothed"]
            gesture = hand_data["gesture"]
            
            # Pointer avec le bout de l'index (landmark 8)
            idx_x = int(smoothed[8][0] * w)
            idx_y = int(smoothed[8][1] * h)

            # Hover Bouton 1 (Snake)
            if btn1_x1 <= idx_x <= btn1_x2 and y1 <= idx_y <= y2:
                hover1 = True
                if gesture.gesture_type == GestureType.PINCH:
                    is_currently_pinching = True
                    if not block_clicks and not self.was_pinching:
                        action_triggered = 1

            # Hover Bouton 2 (Pong)
            if btn2_x1 <= idx_x <= btn2_x2 and y1 <= idx_y <= y2:
                hover2 = True
                if gesture.gesture_type == GestureType.PINCH:
                    is_currently_pinching = True
                    if not block_clicks and not self.was_pinching:
                        action_triggered = 2

            # Dessiner le curseur néon
            cursor_color = (0, 255, 128) if gesture.gesture_type != GestureType.PINCH else (0, 0, 255)
            cursor_radius = 12 if gesture.gesture_type != GestureType.PINCH else 6
            cv2.circle(frame, (idx_x, idx_y), cursor_radius + 4, cursor_color, 1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (idx_x, idx_y), cursor_radius, cursor_color, -1, lineType=cv2.LINE_AA)

        self.was_pinching = is_currently_pinching

        # ── 4. Rendu des Boutons ──
        # Bouton 1 (Snake)
        color1 = (0, 255, 128) if hover1 else (0, 255, 255)
        bg_opacity1 = 0.25 if hover1 else 0.08
        btn1_overlay = frame.copy()
        cv2.rectangle(btn1_overlay, (btn1_x1, y1), (btn1_x2, y2), color1, -1)
        cv2.addWeighted(btn1_overlay, bg_opacity1, frame, 1.0 - bg_opacity1, 0, frame)
        cv2.rectangle(frame, (btn1_x1, y1), (btn1_x2, y2), color1, 2 if not hover1 else 3, lineType=cv2.LINE_AA)

        # Bouton 2 (Pong)
        color2 = (0, 255, 128) if hover2 else (255, 50, 255)
        bg_opacity2 = 0.25 if hover2 else 0.08
        btn2_overlay = frame.copy()
        cv2.rectangle(btn2_overlay, (btn2_x1, y1), (btn2_x2, y2), color2, -1)
        cv2.addWeighted(btn2_overlay, bg_opacity2, frame, 1.0 - bg_opacity2, 0, frame)
        cv2.rectangle(frame, (btn2_x1, y1), (btn2_x2, y2), color2, 2 if not hover2 else 3, lineType=cv2.LINE_AA)

        # ── 5. Rendu du Texte des Boutons ──
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Bouton 1 (Snake)
        cv2.putText(frame, "1. JEU : SNAKE", (btn1_x1 + 25, y1 + 50), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Guidez le serpent avec votre index.", (btn1_x1 + 25, y1 + 95), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Evitez les murs et votre queue !", (btn1_x1 + 25, y1 + 120), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "- Physique de vitesse progressive", (btn1_x1 + 25, y1 + 170), font, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(frame, "- Particules néons lumineuses", (btn1_x1 + 25, y1 + 195), font, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
        
        shortcut1_color = (0, 255, 255) if hover1 else (150, 150, 150)
        cv2.putText(frame, "[ Clavier : 1 ]", (btn1_x1 + 25, y2 - 30), font, 0.45, shortcut1_color, 1, cv2.LINE_AA)

        # Bouton 2 (Pong)
        cv2.putText(frame, "2. JEU : PONG vs IA", (btn2_x1 + 25, y1 + 50), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Affrontez l'intelligence artificielle.", (btn2_x1 + 25, y1 + 95), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Controlex la raquette du bas.", (btn2_x1 + 25, y1 + 120), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "- Système de 5 Power-ups", (btn2_x1 + 25, y1 + 170), font, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(frame, "- IA reactive a vitesse variable", (btn2_x1 + 25, y1 + 195), font, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

        shortcut2_color = (0, 255, 255) if hover2 else (150, 150, 150)
        cv2.putText(frame, "[ Clavier : 2 ]", (btn2_x1 + 25, y2 - 30), font, 0.45, shortcut2_color, 1, cv2.LINE_AA)

        # ── 6. Titre de la section ──
        title = "SELECTION DU JEU"
        cv2.putText(frame, title, (w // 2 - 200, 110), font, 1.4, (255, 50, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, title, (w // 2 - 200, 110), font, 1.4, (255, 255, 255), 2, cv2.LINE_AA)

        subtitle = "MODULE DE DIVERTISSEMENT INTERACTIF"
        cv2.putText(frame, subtitle, (w // 2 - 180, 145), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Coins de titre
        bracket_y = 60
        bracket_h = 100
        bracket_x1 = w // 2 - 280
        bracket_x2 = w // 2 + 280
        cv2.line(frame, (bracket_x1, bracket_y), (bracket_x1 + 20, bracket_y), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (bracket_x1, bracket_y), (bracket_x1, bracket_y + bracket_h), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (bracket_x1, bracket_y + bracket_h), (bracket_x1 + 20, bracket_y + bracket_h), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (bracket_x2, bracket_y), (bracket_x2 - 20, bracket_y), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (bracket_x2, bracket_y), (bracket_x2, bracket_y + bracket_h), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (bracket_x2, bracket_y + bracket_h), (bracket_x2 - 20, bracket_y + bracket_h), (0, 255, 255), 2, cv2.LINE_AA)

        # ── 7. Pied de page ──
        cv2.putText(frame, "Pointez avec l'index pour survoler  |  Pincez (pouce + index) pour valider", 
                    (w // 2 - 340, h - 50), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "[ Raccourci clavier : ESC ou M pour retourner au Menu Principal ]", 
                    (w // 2 - 290, h - 25), font, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

        return frame, action_triggered
