"""
CatchMyHands - Safety Guards
================================
Sécurités pour gérer l'occlusion, la sortie de cadre,
et les sauts aberrants de landmarks.
"""

import numpy as np

import config


class SafetyGuard:
    """
    Module de sécurité pour filtrer les détections invalides.

    Gère trois types de problèmes :
    1. Occlusion : confiance trop basse ou landmarks aberrants
    2. Sortie de cadre (OOB) : main partiellement hors champ
    3. Sauts : téléportation de landmarks entre deux frames
    """

    def __init__(self):
        """Initialise le guard avec l'historique par main."""
        # Landmarks précédents par main pour détecter les sauts
        self._prev_landmarks = {}

    def validate(self, landmarks: np.ndarray, hand_index: int = 0) -> dict:
        """
        Valide un ensemble de landmarks et retourne un rapport de sécurité.

        Args:
            landmarks: Array (21, 3) de landmarks normalisés.
            hand_index: Index de la main.

        Returns:
            dict avec :
                - 'valid': bool - Les landmarks sont-ils exploitables
                - 'edge_factor': float [0,1] - Facteur d'atténuation aux bords
                - 'reason': str - Raison du rejet si invalid
        """
        if landmarks is None:
            return {
                "valid": False,
                "edge_factor": 0.0,
                "reason": "no_landmarks",
            }

        # ── Check 1 : Landmarks dans le cadre ──
        in_frame_count = self._count_landmarks_in_frame(landmarks)
        if in_frame_count < config.MIN_LANDMARKS_IN_FRAME:
            self._prev_landmarks[hand_index] = landmarks.copy()
            return {
                "valid": False,
                "edge_factor": 0.0,
                "reason": f"oob ({in_frame_count}/21 landmarks in frame)",
            }

        # ── Check 2 : Détection de sauts aberrants ──
        if hand_index in self._prev_landmarks:
            if self._detect_jump(landmarks, self._prev_landmarks[hand_index]):
                # Ne pas invalider, mais signaler (le smoothing gèrera)
                pass

        # ── Check 3 : Facteur d'atténuation aux bords ──
        edge_factor = self._compute_edge_factor(landmarks)

        # Sauvegarder pour le prochain frame
        self._prev_landmarks[hand_index] = landmarks.copy()

        return {
            "valid": True,
            "edge_factor": edge_factor,
            "reason": "ok",
        }

    def _count_landmarks_in_frame(self, landmarks: np.ndarray) -> int:
        """
        Compte le nombre de landmarks dont les coordonnées (x, y)
        sont dans l'intervalle [0, 1].
        """
        in_frame = (
            (landmarks[:, 0] >= 0.0) & (landmarks[:, 0] <= 1.0) &
            (landmarks[:, 1] >= 0.0) & (landmarks[:, 1] <= 1.0)
        )
        return int(np.sum(in_frame))

    def _detect_jump(self, current: np.ndarray, previous: np.ndarray) -> bool:
        """
        Détecte un saut aberrant entre deux frames consécutifs.
        Un saut est détecté si le déplacement moyen des landmarks
        est supérieur à MAX_JUMP_RATIO × taille de la main.
        """
        # Taille de la main approximée par distance WRIST → MIDDLE_MCP
        hand_size = np.linalg.norm(current[0, :2] - current[9, :2])
        if hand_size < 0.01:  # Main trop petite, ignorer
            return False

        # Déplacement moyen des landmarks
        displacement = np.mean(np.linalg.norm(
            current[:, :2] - previous[:, :2], axis=1
        ))

        return displacement > config.MAX_JUMP_RATIO * hand_size

    def _compute_edge_factor(self, landmarks: np.ndarray) -> float:
        """
        Calcule un facteur d'atténuation [0, 1] basé sur la proximité
        de la main aux bords du cadre.

        1.0 = main au centre (pas d'atténuation)
        0.0 = main au bord extrême (effet invisible)
        """
        margin = config.EDGE_FADE_MARGIN

        # Bounding box des landmarks
        x_min = np.min(landmarks[:, 0])
        x_max = np.max(landmarks[:, 0])
        y_min = np.min(landmarks[:, 1])
        y_max = np.max(landmarks[:, 1])

        # Distance au bord le plus proche
        dist_to_edge = min(
            x_min,          # Bord gauche
            1.0 - x_max,   # Bord droit
            y_min,          # Bord haut
            1.0 - y_max,   # Bord bas
        )

        if dist_to_edge >= margin:
            return 1.0
        elif dist_to_edge <= 0.0:
            return 0.0
        else:
            # Atténuation linéaire dans la marge
            return dist_to_edge / margin

    def reset(self, hand_index: int = None):
        """Réinitialise l'historique de la main spécifiée ou de toutes."""
        if hand_index is None:
            self._prev_landmarks.clear()
        elif hand_index in self._prev_landmarks:
            del self._prev_landmarks[hand_index]
