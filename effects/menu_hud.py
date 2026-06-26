"""
CatchMyHands - Menu HUD Effect
================================
Affiche un menu de contrôle sur le côté droit de la fenêtre (style cyberpunk).
Permet de visualiser l'état des filtres et options du système en temps réel.
"""

import cv2
import numpy as np
import config

class MenuHUDEffect:
    """
    Gère l'affichage d'un menu HUD interactif sur le côté droit de la vidéo.
    Affiche le statut des filtres configurables via le clavier (ex. touche 1, 2, 3).
    """

    def __init__(self):
        pass

    def render(self, frame: np.ndarray, option_bw_active: bool) -> np.ndarray:
        """
        Dessine le panneau du menu et ses options sur l'image.

        Args:
            frame: Image BGR d'OpenCV.
            option_bw_active: Statut de l'Option 1 (Noir et Blanc).

        Returns:
            L'image avec le menu HUD appliqué.
        """
        h, w = frame.shape[:2]
        menu_w = config.MENU_WIDTH
        
        # Coordonnées du conteneur du menu (marge de 15px des bords)
        margin = 15
        x1 = w - menu_w - margin
        x2 = w - margin
        y1 = 45  # En dessous du HUD FPS (qui fait 35px)
        y2 = h - margin

        if x1 < 0 or y1 >= y2:
            return frame

        # ── 1. Fond semi-transparent ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 10, 15), -1)
        cv2.addWeighted(overlay, config.MENU_BG_OPACITY, frame, 1.0 - config.MENU_BG_OPACITY, 0, frame)

        # ── 2. Bordures néon (Effet de lueur et cadre) ──
        # Lueur externe
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.MENU_BORDER_COLOR, 3, lineType=cv2.LINE_AA)
        # Bordure interne blanche
        cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # Éléments décoratifs aux angles (style cyberpunk)
        bracket_len = 12
        # Angle haut-gauche
        cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), (255, 255, 255), 2, cv2.LINE_AA)
        # Angle bas-droit
        cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), (255, 255, 255), 2, cv2.LINE_AA)

        # ── 3. Contenu textuel ──
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Titre du menu
        title = "SYS. CONTROL"
        cv2.putText(frame, title, (x1 + 15, y1 + 30), font, 0.6, config.MENU_TITLE_COLOR, 2, cv2.LINE_AA)
        
        # Ligne de séparation sous le titre
        cv2.line(frame, (x1 + 15, y1 + 45), (x2 - 15, y1 + 45), config.MENU_BORDER_COLOR, 1, cv2.LINE_AA)

        # Liste des options
        y_offset = y1 + 80
        line_spacing = 40

        # --- OPTION 1 : SCANNER B&W ---
        opt1_key = "[1]"
        opt1_name = "Scanner B&W"
        cv2.putText(frame, opt1_key, (x1 + 15, y_offset), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, opt1_name, (x1 + 45, y_offset), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        status1_text = "ON" if option_bw_active else "OFF"
        status1_color = config.MENU_COLOR_ACTIVE if option_bw_active else config.MENU_COLOR_INACTIVE
        cv2.putText(frame, status1_text, (x2 - 45, y_offset), font, 0.45, status1_color, 2, cv2.LINE_AA)

        # --- OPTION 2 : OPTION VIDE (LOCKED) ---
        y_offset += line_spacing
        opt2_key = "[2]"
        opt2_name = "Option 2"
        cv2.putText(frame, opt2_key, (x1 + 15, y_offset), font, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, opt2_name, (x1 + 45, y_offset), font, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, "LOCKED", (x2 - 75, y_offset), font, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

        # --- OPTION 3 : OPTION VIDE (LOCKED) ---
        y_offset += line_spacing
        opt3_key = "[3]"
        opt3_name = "Option 3"
        cv2.putText(frame, opt3_key, (x1 + 15, y_offset), font, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, opt3_name, (x1 + 45, y_offset), font, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, "LOCKED", (x2 - 75, y_offset), font, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

        # --- SECTION INFO BAS ---
        cv2.line(frame, (x1 + 15, y2 - 55), (x2 - 15, y2 - 55), (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(frame, "TOGGLES : 1, 2, 3", (x1 + 15, y2 - 40), font, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, "PINCH BOTH HANDS", (x1 + 15, y2 - 25), font, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, "TO ACTIVATE FRAME", (x1 + 15, y2 - 10), font, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

        return frame
