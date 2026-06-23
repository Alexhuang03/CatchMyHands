"""
CatchMyHands - Smoothing Filter
===================================
Filtre Exponential Moving Average (EMA) appliqué
par landmark pour réduire le jittering (tremblement).
"""

import numpy as np

import config


class LandmarkSmoother:
    """
    Filtre EMA (Exponential Moving Average) par main.

    Applique un lissage indépendant sur chaque coordonnée (x, y, z)
    de chaque landmark pour réduire le bruit de détection.

    Formule :
        smoothed[t] = α × raw[t] + (1 - α) × smoothed[t-1]

    α proche de 1.0 → suit le raw (réactif, mais tremblant)
    α proche de 0.0 → très lissé (stable, mais latent)
    """

    def __init__(self, alpha: float = None):
        """
        Args:
            alpha: Facteur de lissage (0.0 à 1.0).
                   Utilise config.SMOOTHING_FACTOR si non spécifié.
        """
        self.alpha = alpha if alpha is not None else config.SMOOTHING_FACTOR

        # État du filtre par main : {hand_index: (smoothed_landmarks, frames_missing)}
        self._states = {}

    def smooth(self, landmarks: np.ndarray, hand_index: int = 0) -> np.ndarray:
        """
        Applique le lissage EMA sur les landmarks d'une main.

        Args:
            landmarks: Array (21, 3) de landmarks bruts.
            hand_index: Index de la main pour le suivi indépendant.

        Returns:
            Array (21, 3) de landmarks lissés.
        """
        if landmarks is None:
            # Si pas de landmarks, incrémenter le compteur de frames manquantes
            if hand_index in self._states:
                self._states[hand_index] = (
                    self._states[hand_index][0],
                    self._states[hand_index][1] + 1,
                )
                # Reset si trop de frames sans détection
                if self._states[hand_index][1] >= config.SMOOTHING_RESET_FRAMES:
                    del self._states[hand_index]
            return None

        if hand_index not in self._states:
            # Première détection : pas de lissage, on initialise
            self._states[hand_index] = (landmarks.copy(), 0)
            return landmarks.copy()

        prev_smoothed, _ = self._states[hand_index]

        # EMA : smoothed = α × raw + (1 - α) × prev_smoothed
        smoothed = self.alpha * landmarks + (1.0 - self.alpha) * prev_smoothed

        self._states[hand_index] = (smoothed.copy(), 0)
        return smoothed

    def reset(self, hand_index: int = None):
        """
        Réinitialise le filtre.

        Args:
            hand_index: Index de la main à reset.
                        Si None, reset toutes les mains.
        """
        if hand_index is None:
            self._states.clear()
        elif hand_index in self._states:
            del self._states[hand_index]

    def set_alpha(self, alpha: float):
        """
        Modifie le facteur de lissage en temps réel.

        Args:
            alpha: Nouveau facteur (0.0 à 1.0).
        """
        self.alpha = max(0.0, min(1.0, alpha))
