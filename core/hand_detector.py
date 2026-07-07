"""
CatchMyHands - Hand Detector
==============================
Wrapper autour de MediaPipe Tasks HandLandmarker.
Gère le téléchargement du modèle, l'initialisation en mode LIVE_STREAM,
et l'exposition thread-safe des résultats.
"""

import os
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

import config


class HandDetector:
    """
    Détecteur de mains basé sur MediaPipe HandLandmarker (Tasks API).

    Utilise le mode LIVE_STREAM avec callback asynchrone pour maintenir
    un FPS élevé sans bloquer la boucle principale.
    """

    # Mapping lisible des 21 landmarks
    LANDMARK_NAMES = {
        0: "WRIST",
        1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
        5: "INDEX_MCP", 6: "INDEX_PIP", 7: "INDEX_DIP", 8: "INDEX_TIP",
        9: "MIDDLE_MCP", 10: "MIDDLE_PIP", 11: "MIDDLE_DIP", 12: "MIDDLE_TIP",
        13: "RING_MCP", 14: "RING_PIP", 15: "RING_DIP", 16: "RING_TIP",
        17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP",
    }

    # Indices pratiques des bouts de doigts
    FINGER_TIPS = [4, 8, 12, 16, 20]
    FINGER_PIPS = [3, 6, 10, 14, 18]
    FINGER_MCPS = [2, 5, 9, 13, 17]

    def __init__(self):
        """Initialise le détecteur : télécharge le modèle et configure MediaPipe."""
        self._ensure_model_exists()

        self._result = None
        self._result_lock = threading.Lock()
        self._timestamp_ms = 0

        # Aliases pour l'API Tasks
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Configuration du délégué (GPU / CPU)
        delegate = BaseOptions.Delegate.GPU if config.MODEL_DELEGATE == "GPU" else BaseOptions.Delegate.CPU

        try:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=config.MODEL_PATH, delegate=delegate),
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_hands=config.NUM_HANDS,
                min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=config.MIN_PRESENCE_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
                result_callback=self._on_result,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
            print(f"[HandDetector] Initialisé avec accélération : {config.MODEL_DELEGATE}")
        except Exception as e:
            if delegate == BaseOptions.Delegate.GPU:
                print(f"[HandDetector] ⚠ Échec d'activation GPU ({e}). Repli sur le CPU...")
                options = HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=config.MODEL_PATH, delegate=BaseOptions.Delegate.CPU),
                    running_mode=VisionRunningMode.LIVE_STREAM,
                    num_hands=config.NUM_HANDS,
                    min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                    min_hand_presence_confidence=config.MIN_PRESENCE_CONFIDENCE,
                    min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
                    result_callback=self._on_result,
                )
                self._landmarker = HandLandmarker.create_from_options(options)
                print("[HandDetector] Initialisé avec accélération : CPU (Fallback)")
            else:
                raise e

    def _ensure_model_exists(self):
        """Télécharge le modèle .task si absent du répertoire assets."""
        if os.path.exists(config.MODEL_PATH):
            return

        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
        print(f"[HandDetector] Téléchargement du modèle depuis {config.MODEL_URL}...")

        try:
            urllib.request.urlretrieve(config.MODEL_URL, config.MODEL_PATH)
            file_size_mb = os.path.getsize(config.MODEL_PATH) / (1024 * 1024)
            print(f"[HandDetector] Modèle téléchargé ({file_size_mb:.1f} MB)")
        except Exception as e:
            raise RuntimeError(
                f"Impossible de télécharger le modèle MediaPipe : {e}\n"
                f"Téléchargez-le manuellement depuis {config.MODEL_URL}\n"
                f"et placez-le dans {config.MODEL_PATH}"
            )

    def _on_result(self, result, output_image, timestamp_ms):
        """
        Callback appelé par MediaPipe à chaque résultat d'inférence.
        Thread-safe : protégé par un lock.
        """
        with self._result_lock:
            self._result = result

    def detect_async(self, frame_bgr, timestamp_ms):
        """
        Lance une détection asynchrone sur un frame BGR (format OpenCV).

        Args:
            frame_bgr: Frame BGR (numpy array) depuis OpenCV VideoCapture.
            timestamp_ms: Timestamp en millisecondes (doit être croissant).
        """
        # Convertir BGR → RGB pour MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Envoyer au détecteur (non-bloquant)
        # Si le détecteur est occupé, le frame est ignoré silencieusement
        try:
            self._landmarker.detect_async(mp_image, timestamp_ms)
        except Exception:
            # MediaPipe peut lever une erreur si le timestamp n'est pas croissant
            pass

    def get_result(self):
        """
        Récupère le dernier résultat de détection.

        Returns:
            HandLandmarkerResult ou None si aucune détection.
        """
        with self._result_lock:
            return self._result

    def get_landmarks_array(self, hand_index=0):
        """
        Extrait les landmarks d'une main sous forme de numpy array.

        Args:
            hand_index: Index de la main (0 = première, 1 = seconde).

        Returns:
            numpy array de shape (21, 3) avec (x, y, z) normalisés,
            ou None si la main n'est pas détectée.
        """
        result = self.get_result()
        if result is None or hand_index >= len(result.hand_landmarks):
            return None

        landmarks = result.hand_landmarks[hand_index]
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    def get_handedness(self, hand_index=0):
        """
        Récupère la latéralité d'une main ("Left" ou "Right").

        Args:
            hand_index: Index de la main.

        Returns:
            str "Left" ou "Right", ou None si non détecté.
        """
        result = self.get_result()
        if result is None or hand_index >= len(result.handedness):
            return None

        label = result.handedness[hand_index][0].category_name
        # Inverser car le frame est flippé horizontalement (miroir)
        # MediaPipe détecte sur l'image originale, donc Left ↔ Right
        if label == "Left":
            return "Right"
        elif label == "Right":
            return "Left"
        return label

    def get_num_hands_detected(self):
        """Retourne le nombre de mains actuellement détectées."""
        result = self.get_result()
        if result is None:
            return 0
        return len(result.hand_landmarks)

    def close(self):
        """Libère les ressources MediaPipe."""
        if self._landmarker:
            self._landmarker.close()
            print("[HandDetector] Ressources libérées")
