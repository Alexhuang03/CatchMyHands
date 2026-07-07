"""
CatchMyHands - Menu HUD Effect
================================
Affiche un menu de contrôle sur le côté droit de la fenêtre (style cyberpunk).
Permet de visualiser l'état des filtres et options du système en temps réel.
Affiche l'attribution dynamique des filtres aux doigts.
"""

import cv2
import numpy as np
import config

class MenuHUDEffect:
    """
    Gère l'affichage d'un menu HUD interactif sur le côté droit de la vidéo.
    Affiche le statut des filtres configurables et leur attribution aux doigts.
    """

    # Icônes des doigts pour l'affichage
    FINGER_ICONS = ["Index", "Majeur", "Annul.", "Auric."]

    def __init__(self):
        pass

    def render(self, frame: np.ndarray, active_filters_ordered: list) -> np.ndarray:
        """
        Dessine le panneau du menu et ses options sur l'image.

        Args:
            frame: Image BGR d'OpenCV.
            active_filters_ordered: Liste ordonnée des filtres activés (ex: ["bw", "pixelate"]).

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

        # ── Section : Filtres disponibles ──
        y_offset = y1 + 70
        line_spacing = 30

        # Sous-titre
        cv2.putText(frame, "FILTRES", (x1 + 15, y_offset), font, 0.4, config.MENU_BORDER_COLOR, 1, cv2.LINE_AA)
        y_offset += 8
        cv2.line(frame, (x1 + 15, y_offset), (x1 + 75, y_offset), (60, 60, 60), 1, cv2.LINE_AA)
        y_offset += line_spacing

        # Construire un set pour savoir quels filtres sont actifs
        active_set = set(active_filters_ordered)

        # Afficher chaque filtre disponible avec sa touche et son état
        all_filters = config.AVAILABLE_FILTERS
        for idx, filter_name in enumerate(all_filters):
            key_num = idx + 1
            display_name = config.FILTER_DISPLAY_NAMES.get(filter_name, filter_name)
            is_active = filter_name in active_set

            # Touche
            key_text = f"[{key_num}]"
            cv2.putText(frame, key_text, (x1 + 15, y_offset), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # Nom du filtre
            cv2.putText(frame, display_name, (x1 + 45, y_offset), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

            # Statut ON/OFF
            status_text = "ON" if is_active else "OFF"
            status_color = config.MENU_COLOR_ACTIVE if is_active else config.MENU_COLOR_INACTIVE
            cv2.putText(frame, status_text, (x2 - 45, y_offset), font, 0.4, status_color, 2, cv2.LINE_AA)

            y_offset += line_spacing

        # ── Section : Attribution aux doigts ──
        y_offset += 10
        cv2.line(frame, (x1 + 15, y_offset), (x2 - 15, y_offset), (60, 60, 60), 1, cv2.LINE_AA)
        y_offset += 25

        cv2.putText(frame, "DOIGTS", (x1 + 15, y_offset), font, 0.4, config.MENU_BORDER_COLOR, 1, cv2.LINE_AA)
        y_offset += 8
        cv2.line(frame, (x1 + 15, y_offset), (x1 + 75, y_offset), (60, 60, 60), 1, cv2.LINE_AA)
        y_offset += line_spacing

        finger_keys = ["index", "middle", "ring", "pinky"]

        for i in range(4):
            finger_name = self.FINGER_ICONS[i]
            finger_key = finger_keys[i]
            color = config.FINGER_FILTER_COLORS[finger_key]

            # Indicateur coloré (petit carré)
            sq_x = x1 + 15
            sq_y = y_offset - 8
            cv2.rectangle(frame, (sq_x, sq_y), (sq_x + 8, sq_y + 8), color, -1, cv2.LINE_AA)

            # Nom du doigt
            cv2.putText(frame, finger_name, (x1 + 30, y_offset), font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

            # Filtre attribué ou "---"
            if i < len(active_filters_ordered):
                assigned_filter = active_filters_ordered[i]
                assigned_name = config.FILTER_DISPLAY_NAMES.get(assigned_filter, assigned_filter)
                # Tronquer si trop long
                if len(assigned_name) > 10:
                    assigned_name = assigned_name[:9] + "."
                cv2.putText(frame, assigned_name, (x1 + 95, y_offset), font, 0.38, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "---", (x1 + 95, y_offset), font, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

            y_offset += line_spacing

        # ── Pied de page ──
        cv2.line(frame, (x1 + 15, y2 - 55), (x2 - 15, y2 - 55), (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(frame, "TOGGLES : 1, 2, 3, 4", (x1 + 15, y2 - 40), font, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, "L-SHAPE BOTH HANDS", (x1 + 15, y2 - 25), font, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, "TO ACTIVATE FRAME", (x1 + 15, y2 - 10), font, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

        return frame
