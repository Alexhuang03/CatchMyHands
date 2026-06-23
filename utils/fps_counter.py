"""
CatchMyHands - FPS Counter
==============================
Compteur de frames par seconde lissé avec fenêtre glissante.
"""

import time
from collections import deque


class FPSCounter:
    """
    Compteur FPS basé sur une fenêtre glissante.

    Plus précis qu'un simple 1/dt car il moyenne sur N frames,
    ce qui donne un affichage stable et lisible.
    """

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Nombre de frames pour le calcul de la moyenne.
        """
        self._timestamps = deque(maxlen=window_size)
        self._fps = 0.0

    def tick(self):
        """Enregistre un nouveau frame et recalcule le FPS."""
        now = time.perf_counter()
        self._timestamps.append(now)

        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self._fps = (len(self._timestamps) - 1) / elapsed

    @property
    def fps(self) -> float:
        """FPS actuel (lissé sur la fenêtre)."""
        return self._fps

    @property
    def fps_str(self) -> str:
        """FPS formaté pour l'affichage HUD."""
        return f"{self._fps:.0f} FPS"
