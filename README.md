# CatchMyHands 

> Système de Vision par Ordinateur Interactive en temps réel — Détection des mains, reconnaissance de gestes et effets de filtres immersifs en plein écran.

---

## 🎮 Concept

**CatchMyHands** utilise **MediaPipe Hands** (Google AI Edge) pour détecter les 21 points de repère de chaque main en temps réel. Lorsque les deux mains forment un geste en L (pouces et index étendus, autres doigts repliés), elles créent un **cadre néon dynamique** dont les 4 coins se situent directement sur le bout des index et des pouces.

Le système est entièrement **robuste aux rotations** (fonctionne en paysage, portrait, diagonal) et trie géométriquement les sommets par rapport à leur centre de gravité. Cela empêche le cadre de se croiser (effet sablier) ou de se briser lorsque vos mains sont inversées (par exemple, pouce droit vers le haut et index droit vers le bas).

À l'intérieur de ce cadre, plusieurs filtres avancés peuvent être appliqués et combinés à l'aide de raccourcis clavier.

---

## ✨ Fonctionnalités (Phase 1 — PoC)

### 🤏 Gestes & Effets

| Geste / Touche | Action / Effet |
| :--- | :--- |
| **Main ouverte** | Affiche une aura d'énergie cyan pulsante au centre de la paume (atténuation progressive aux bords). |
| **Geste en L (Cadre)** | Forme un cadre néon reliant les index et les pouces des deux mains (4 coins définis directement par le bout des doigts). Les effets d'aura et de squelette sont masqués pendant l'activation. |
| Touche **`1`** (ou `&`) | Active/Désactive l'**Option 1 : Scanner B&W** (convertit l'intérieur du cadre en noir et blanc). |
| Touche **`2`** (ou `é`) | Active/Désactive l'**Option 2 : Pixelate** (effet mosaïque rétro pixelisé). |

### 🛠️ Bonus Techniques & Affichage
- **Lancement Plein Écran** : L'application s'ouvre par défaut en plein écran (`cv2.WINDOW_FULLSCREEN`) pour une immersion totale.
- **HUD Cyberpunk (SYS. CONTROL)** : Un panneau latéral droit transparent affiche l'état actif/inactif de chaque filtre ainsi que les instructions de contrôle.
- **Lissage EMA Anti-Jittering** : Les repères des mains sont lissés pour éliminer tout tremblement lié à la webcam.
- **Squelette Debug** : Affichage optionnel de la structure osseuse de la main avec codes couleurs distincts par doigt.

---

## 📦 Installation

```bash
# Cloner le projet
git clone https://github.com/Alexhuang03/CatchMyHands.git
cd CatchMyHands

# Créer un environnement virtuel (recommandé)
python -m venv venv
venv\Scripts\activate  # Windows
# ou: source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

Le modèle MediaPipe `hand_landmarker.task` (~12 Mo) sera téléchargé automatiquement dans le répertoire `assets/` lors du premier lancement.

---

## 🚀 Utilisation

```bash
python main.py
```

### Contrôles clavier

| Touche | Action |
| :--- | :--- |
| **`Q`** / **`ESC`** | Quitter l'application |
| **`1`** / **`&`** | Activer/Désactiver l'Option 1 (Scanner B&W) |
| **`2`** / **`é`** | Activer/Désactiver l'Option 2 (Pixelate) |
| **`D`** | Afficher/Masquer le squelette des mains (debug) |
| **`I`** | Afficher/Masquer les IDs des landmarks |
| **`S`** | Prendre une capture d'écran (sauvegardée dans `screenshots/`) |
| **`+`** / **`-`** | Ajuster la sensibilité de détection du pincement |

---

## 📂 Architecture du Projet

```
CatchMyHands/
├── main.py              # Point d'entrée & boucle principale OpenCV/MediaPipe
├── config.py            # Paramètres de configuration (FPS, couleurs, seuils)
├── core/
│   ├── hand_detector.py # Wrapper MediaPipe HandLandmarker (Task API)
│   ├── gesture_engine.py# Calculs de géométrie et classification des gestes
│   ├── smoothing.py     # Lissage par moyenne mobile exponentielle (EMA)
│   └── safety.py        # Validations de sécurité (sortie de cadre, jumps de repères)
├── effects/
│   ├── box_frame.py     # Effet de cadre néon et application des filtres intérieurs
│   ├── menu_hud.py      # HUD latéral cyberpunk affichant le statut des filtres
│   ├── minecraft_effect.py # Effet de pixelisation (Pixelate)
│   ├── overlay.py       # Effet d'aura sur les mains ouvertes
│   └── skeleton.py      # Rendu du squelette de debug des doigts
├── utils/
│   └── fps_counter.py   # Compteur et lissage des images par seconde (FPS)
├── assets/
│   └── hand_landmarker.task # Modèle de tracking IA MediaPipe
└── screenshots/         # Dossier des captures d'écran capturées par la touche S
```

---

## 🗺️ Roadmap

- [x] **Phase 1** : PoC local interactif (OpenCV + MediaPipe, Cadre Néon, Filtres combinables)
- [ ] **Phase 2** : Conteneurisation Docker
- [ ] **Phase 3** : Optimisation et déploiement Edge (Raspberry Pi)
- [ ] **Phase 4** : Interface Web multi-utilisateurs (WebSockets/WebRTC)
