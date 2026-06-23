"""
CatchMyHands - Configuration centralisée
==========================================
Tous les paramètres ajustables du système regroupés ici.
Modifier ces valeurs pour adapter le comportement sans toucher au code.
"""

# ──────────────────────────────────────────
# 📹 Caméra
# ──────────────────────────────────────────
CAMERA_INDEX = 0           # Index de la caméra (0 = webcam par défaut)
CAMERA_WIDTH = 640         # Largeur de capture en pixels
CAMERA_HEIGHT = 480        # Hauteur de capture en pixels

# ──────────────────────────────────────────
# 🧠 MediaPipe HandLandmarker
# ──────────────────────────────────────────
MODEL_PATH = "assets/hand_landmarker.task"  # Chemin vers le modèle
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
NUM_HANDS = 2                          # Nombre max de mains détectées
MIN_DETECTION_CONFIDENCE = 0.7         # Seuil confiance détection paume
MIN_PRESENCE_CONFIDENCE = 0.5          # Seuil confiance présence main
MIN_TRACKING_CONFIDENCE = 0.5          # Seuil confiance tracking

# ──────────────────────────────────────────
# 🤏 Détection de gestes — Seuils
# ──────────────────────────────────────────
PINCH_THRESHOLD = 0.05                 # Distance normalisée max pour pincement
PINCH_RELEASE_THRESHOLD = 0.07         # Seuil de relâchement (hystérésis)
OPEN_HAND_MIN_EXTENSION = 0.15         # Extension min d'un doigt (vs wrist)
FIST_MAX_EXTENSION = 0.08             # Extension max pour poing fermé (clear canvas)

# ──────────────────────────────────────────
# 🔧 Lissage (Anti-Jittering)
# ──────────────────────────────────────────
SMOOTHING_FACTOR = 0.6                 # Alpha EMA (0=très lissé, 1=raw data)
SMOOTHING_RESET_FRAMES = 5            # Frames sans détection avant reset du filtre

# ──────────────────────────────────────────
# 🛡️ Sécurités
# ──────────────────────────────────────────
MIN_LANDMARKS_IN_FRAME = 15           # Min landmarks dans le cadre (sur 21)
MAX_JUMP_RATIO = 0.3                  # Saut max entre 2 frames (ratio taille main)
EDGE_FADE_MARGIN = 0.1                # Marge pour atténuation progressive aux bords

# ──────────────────────────────────────────
# 🎨 Effets visuels — Dessin (Pincement)
# ──────────────────────────────────────────
DRAW_COLOR = (0, 255, 128)            # Couleur BGR du trait (vert néon)
DRAW_THICKNESS_MIN = 2                # Épaisseur min du trait
DRAW_THICKNESS_MAX = 8                # Épaisseur max (vitesse lente)
DRAW_FADE_ENABLED = True              # Activer le fade progressif des traits
DRAW_FADE_DURATION = 150              # Nombre de frames avant disparition complète

# ──────────────────────────────────────────
# 🎨 Effets visuels — Overlay Aura (Main ouverte)
# ──────────────────────────────────────────
AURA_IMAGE_PATH = "assets/aura.png"   # Image PNG avec canal alpha
AURA_SCALE_FACTOR = 2.5               # Facteur de taille (vs taille de la main)
AURA_PULSE_SPEED = 0.05               # Vitesse de pulsation (radians/frame)
AURA_PULSE_AMPLITUDE = 0.15           # Amplitude de pulsation (±15%)
AURA_BASE_OPACITY = 0.7               # Opacité de base de l'aura

# ──────────────────────────────────────────
# 🦴 Squelette (Debug)
# ──────────────────────────────────────────
SKELETON_ENABLED_DEFAULT = True        # Afficher le squelette par défaut
SKELETON_POINT_RADIUS = 5             # Rayon des points landmarks
SKELETON_LINE_THICKNESS = 2           # Épaisseur des lignes de connexion
SKELETON_COLORS = {
    "thumb":  (0, 0, 255),             # Rouge
    "index":  (255, 128, 0),           # Bleu-orange
    "middle": (0, 255, 0),             # Vert
    "ring":   (255, 255, 0),           # Cyan
    "pinky":  (255, 0, 255),           # Magenta
    "palm":   (200, 200, 200),         # Gris clair
}

# ──────────────────────────────────────────
# 📊 HUD (Heads-Up Display)
# ──────────────────────────────────────────
HUD_FONT_SCALE = 0.6                  # Taille de police
HUD_COLOR = (255, 255, 255)           # Couleur texte (blanc)
HUD_BG_COLOR = (0, 0, 0)             # Couleur fond
HUD_POSITION = (10, 30)               # Position du FPS counter
