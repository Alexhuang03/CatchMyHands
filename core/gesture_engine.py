"""
CatchMyHands - Gesture Engine
================================
Détection de gestes basée sur la géométrie euclidienne
entre les 21 landmarks de la main.
"""

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np

import config


class GestureType(Enum):
    """Types de gestes reconnus par le moteur."""
    NONE = auto()
    PINCH = auto()        # Pincement pouce-index
    OPEN_HAND = auto()    # Main grande ouverte
    FIST = auto()         # Poing fermé (clear canvas)


@dataclass
class GestureResult:
    """Résultat de la détection de geste pour une main."""
    gesture_type: GestureType = GestureType.NONE
    confidence: float = 0.0
    pinch_position: Optional[Tuple[float, float]] = None  # (x, y) du point de pincement
    palm_center: Optional[Tuple[float, float]] = None      # Centre de la paume
    hand_size: float = 0.0                                  # Taille relative de la main
    pinch_distance: float = 1.0                             # Distance pouce-index


class GestureEngine:
    """
    Moteur de reconnaissance de gestes.

    Analyse les positions relatives des 21 landmarks pour identifier
    des gestes prédéfinis via calcul de distances euclidiennes.
    Implémente l'hystérésis pour éviter les oscillations entre états.
    """

    def __init__(self):
        """Initialise le moteur avec les états de chaque main."""
        # Hystérésis : état précédent par main pour éviter le flickering
        self._prev_gestures = {}

    def analyze(self, landmarks: np.ndarray, hand_index: int = 0) -> GestureResult:
        """
        Analyse les landmarks d'une main pour détecter un geste.

        Args:
            landmarks: Array (21, 3) de landmarks normalisés (x, y, z).
            hand_index: Index de la main (pour le suivi d'hystérésis).

        Returns:
            GestureResult avec le geste détecté et ses métadonnées.
        """
        if landmarks is None or len(landmarks) < 21:
            return GestureResult()

        result = GestureResult()

        # ── Calculs géométriques de base ──
        result.palm_center = self._compute_palm_center(landmarks)
        result.hand_size = self._compute_hand_size(landmarks)

        # ── Détection du pincement ──
        pinch_dist = self._euclidean_2d(landmarks[4], landmarks[8])
        result.pinch_distance = pinch_dist

        # Point milieu entre pouce et index pour la position du trait
        result.pinch_position = (
            (landmarks[4][0] + landmarks[8][0]) / 2.0,
            (landmarks[4][1] + landmarks[8][1]) / 2.0,
        )

        # ── Détection des extensions de doigts ──
        fingers_extended = self._check_fingers_extended(landmarks)
        num_extended = sum(fingers_extended)

        # ── Hystérésis pour le pincement ──
        prev_gesture = self._prev_gestures.get(hand_index, GestureType.NONE)

        # Seuil dynamique avec hystérésis
        if prev_gesture == GestureType.PINCH:
            pinch_threshold = config.PINCH_RELEASE_THRESHOLD
        else:
            pinch_threshold = config.PINCH_THRESHOLD

        # ── Classification du geste ──
        if pinch_dist < pinch_threshold:
            result.gesture_type = GestureType.PINCH
            result.confidence = max(0.0, 1.0 - (pinch_dist / config.PINCH_RELEASE_THRESHOLD))

        elif num_extended >= 4:
            # Main ouverte : au moins 4 doigts étendus (robuste aux variations du pouce)
            result.gesture_type = GestureType.OPEN_HAND
            result.confidence = num_extended / 5.0

        elif num_extended <= 1:
            # Poing fermé : max 1 doigt étendu (robuste aux variations du pouce)
            result.gesture_type = GestureType.FIST
            result.confidence = 1.0 - (num_extended / 5.0)

        else:
            result.gesture_type = GestureType.NONE
            result.confidence = 0.0

        # Mémoriser l'état pour l'hystérésis
        self._prev_gestures[hand_index] = result.gesture_type

        return result

    def check_two_hand_frame(self, left_landmarks: np.ndarray, right_landmarks: np.ndarray) -> bool:
        """
        Détecte si le geste de cadre à deux mains est actif.
        Le cadre est activé si les deux mains font un geste en L (index et pouce étendus,
        autres doigts repliés) en même temps. Implémente une hystérésis robuste.
        """
        if left_landmarks is None or right_landmarks is None:
            self._frame_stable_count = 0
            return False

        # Vérifier si chaque main fait un geste de L
        left_is_l = self._is_l_shape(left_landmarks)
        right_is_l = self._is_l_shape(right_landmarks)

        # Initialiser les attributs d'hystérésis si nécessaire
        if not hasattr(self, '_two_hand_frame_active'):
            self._two_hand_frame_active = False
            self._frame_stable_count = 0

        if self._two_hand_frame_active:
            # Exiger que le geste disparaisse sur au moins une main pendant plusieurs frames pour désactiver
            if left_is_l and right_is_l:
                self._frame_stable_count = 0
            else:
                self._frame_stable_count += 1
                if self._frame_stable_count >= 8:
                    self._two_hand_frame_active = False
                    self._frame_stable_count = 0
        else:
            # Activation quand les deux mains font le L-shape
            if left_is_l and right_is_l:
                self._two_hand_frame_active = True
                self._frame_stable_count = 0

        return self._two_hand_frame_active

    def _is_l_shape(self, landmarks: np.ndarray) -> bool:
        """Vérifie si la main fait un geste de L (index + pouce étendus, au plus 3 doigts étendus au total)."""
        if landmarks is None or len(landmarks) < 21:
            return False
        fingers_extended = self._check_fingers_extended(landmarks)
        num_extended = sum(fingers_extended)
        # Pouce et index doivent être étendus, et pas plus de 3 doigts étendus au total (pour la robustesse au bruit)
        return fingers_extended[0] and fingers_extended[1] and (num_extended <= 3)

    def _euclidean_2d(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Distance euclidienne 2D entre deux landmarks (x, y)."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _euclidean_3d(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Distance euclidienne 3D entre deux landmarks (x, y, z)."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def _compute_palm_center(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Calcule le centre géométrique de la paume.
        Utilise les MCPs (base des doigts) + le poignet.
        """
        palm_indices = [0, 5, 9, 13, 17]  # WRIST + base de chaque doigt
        palm_points = landmarks[palm_indices]
        cx = np.mean(palm_points[:, 0])
        cy = np.mean(palm_points[:, 1])
        return (float(cx), float(cy))

    def _compute_hand_size(self, landmarks: np.ndarray) -> float:
        """
        Estime la taille de la main (distance WRIST → MIDDLE_MCP).
        Utile pour normaliser les seuils et dimensionner les overlays.
        """
        return self._euclidean_2d(landmarks[0], landmarks[9])

    def _check_fingers_extended(self, landmarks: np.ndarray) -> list:
        """
        Vérifie quels doigts sont étendus.
        Utilise les coordonnées 3D pour être parfaitement invariant aux rotations/angles.

        Returns:
            Liste de 5 booléens [pouce, index, majeur, annulaire, auriculaire].
        """
        extended = []

        # ── Pouce ──
        # Le pouce est étendu si le tip (4) est plus éloigné de la base de l'auriculaire (17)
        # que l'articulation IP (3) en 3D.
        thumb_tip_dist = self._euclidean_3d(landmarks[4], landmarks[17])
        thumb_ip_dist = self._euclidean_3d(landmarks[3], landmarks[17])
        extended.append(thumb_tip_dist > thumb_ip_dist)

        # ── Index, Majeur, Annulaire, Auriculaire ──
        # On compare la distance TIP->Poignet vs PIP->Poignet en 3D.
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            dist_tip = self._euclidean_3d(landmarks[tip_idx], landmarks[0])
            dist_pip = self._euclidean_3d(landmarks[pip_idx], landmarks[0])
            # Si le doigt est étendu, son bout (tip) est plus éloigné du poignet que le joint PIP
            extended.append(dist_tip > dist_pip * 1.12)

        return extended
