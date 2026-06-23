"""
CatchMyHands - Skeleton Renderer
====================================
Rendu visuel du squelette de la main (debug et visualisation).
Dessine les 21 landmarks et leurs connexions avec code couleur par doigt.
"""

import cv2
import numpy as np

import config


# Connexions entre landmarks (définies par MediaPipe)
HAND_CONNECTIONS = [
    # Paume
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (5, 9), (9, 13), (13, 17),
    # Pouce
    (1, 2), (2, 3), (3, 4),
    # Index
    (5, 6), (6, 7), (7, 8),
    # Majeur
    (9, 10), (10, 11), (11, 12),
    # Annulaire
    (13, 14), (14, 15), (15, 16),
    # Auriculaire
    (17, 18), (18, 19), (19, 20),
]

# Mapping landmark → groupe de doigt (pour la couleur)
LANDMARK_GROUPS = {
    0: "palm",
    1: "thumb", 2: "thumb", 3: "thumb", 4: "thumb",
    5: "index", 6: "index", 7: "index", 8: "index",
    9: "middle", 10: "middle", 11: "middle", 12: "middle",
    13: "ring", 14: "ring", 15: "ring", 16: "ring",
    17: "pinky", 18: "pinky", 19: "pinky", 20: "pinky",
}


class SkeletonRenderer:
    """
    Rend le squelette de la main sur le flux vidéo.

    Affiche :
    - Les 21 landmarks sous forme de cercles colorés
    - Les connexions entre landmarks sous forme de lignes
    - Optionnellement, les IDs des landmarks
    """

    def __init__(self):
        """Initialise le renderer."""
        self.show_ids = False  # Toggle avec touche clavier

    def render(self, frame: np.ndarray, landmarks: np.ndarray,
               handedness: str = None, edge_factor: float = 1.0) -> np.ndarray:
        """
        Dessine le squelette de la main sur le frame.

        Args:
            frame: Frame BGR.
            landmarks: Array (21, 3) de landmarks normalisés.
            handedness: "Left" ou "Right" (affiché en label).
            edge_factor: Facteur d'atténuation pour les bords [0, 1].

        Returns:
            Frame avec le squelette dessiné.
        """
        if landmarks is None:
            return frame

        h, w = frame.shape[:2]

        # Convertir les coordonnées normalisées en pixels
        points = []
        for lm in landmarks:
            px = int(lm[0] * w)
            py = int(lm[1] * h)
            points.append((px, py))

        # ── Dessiner les connexions ──
        for start_idx, end_idx in HAND_CONNECTIONS:
            p1 = points[start_idx]
            p2 = points[end_idx]

            # Couleur basée sur le groupe du landmark de fin
            group = LANDMARK_GROUPS.get(end_idx, "palm")
            color = config.SKELETON_COLORS.get(group, (200, 200, 200))

            # Appliquer l'atténuation de bord
            if edge_factor < 1.0:
                color = tuple(int(c * edge_factor) for c in color)

            cv2.line(frame, p1, p2, color,
                     config.SKELETON_LINE_THICKNESS,
                     lineType=cv2.LINE_AA)

        # ── Dessiner les landmarks ──
        for i, point in enumerate(points):
            group = LANDMARK_GROUPS.get(i, "palm")
            color = config.SKELETON_COLORS.get(group, (200, 200, 200))

            if edge_factor < 1.0:
                color = tuple(int(c * edge_factor) for c in color)

            # Point extérieur (contour)
            cv2.circle(frame, point, config.SKELETON_POINT_RADIUS + 1,
                       (0, 0, 0), -1, lineType=cv2.LINE_AA)
            # Point intérieur (couleur)
            cv2.circle(frame, point, config.SKELETON_POINT_RADIUS,
                       color, -1, lineType=cv2.LINE_AA)

            # Afficher l'ID du landmark si activé
            if self.show_ids:
                cv2.putText(frame, str(i),
                            (point[0] + 8, point[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # ── Label de latéralité ──
        if handedness:
            wrist = points[0]
            label = f"{handedness}"
            cv2.putText(frame, label,
                        (wrist[0] - 20, wrist[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def toggle_ids(self):
        """Active/désactive l'affichage des IDs de landmarks."""
        self.show_ids = not self.show_ids
        print(f"[Skeleton] IDs {'activés' if self.show_ids else 'désactivés'}")
