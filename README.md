# CatchMyHands 🖐️

> Système de Vision par Ordinateur Interactive en temps réel — Détection des mains et des doigts avec effets visuels AR.

## Concept

Utilise **MediaPipe Hands** (Google AI Edge) pour détecter les 21 points de repère de chaque main en temps réel, puis déclenche des effets visuels interactifs basés sur la reconnaissance de gestes.

## Fonctionnalités (Phase 1 — PoC)

| Geste                               | Effet                                           |
| ----------------------------------- | ----------------------------------------------- |
| **Pincement** (pouce + index) | Dessine des traits colorés à l'écran         |
| **Main ouverte**              | Affiche une aura énergie au centre de la paume |
| **Poing fermé**              | Efface le canvas de dessin                      |

### Bonus techniques

- 🔧 Filtre EMA anti-jittering (lissage des tremblements)
- 🛡️ Sécurité occlusion & sortie de cadre
- 📊 HUD temps réel (FPS, geste détecté, nombre de mains)
- 🦴 Squelette debug avec code couleur par doigt

## 🚀 Installation

```bash
# Cloner le projet
cd CatchMyHands

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

Le modèle MediaPipe (~12 MB) sera téléchargé automatiquement au premier lancement.

## Utilisation

```bash
python main.py
```

### Contrôles clavier

| Touche          | Action                        |
| --------------- | ----------------------------- |
| `Q` / `ESC` | Quitter                       |
| `D`           | Toggle affichage squelette    |
| `I`           | Toggle IDs des landmarks      |
| `C`           | Effacer le canvas de dessin   |
| `S`           | Screenshot                    |
| `+` / `-`   | Ajuster le seuil de pincement |

## Architecture

```
CatchMyHands/
├── main.py              # Boucle principale
├── config.py            # Configuration centralisée
├── core/
│   ├── hand_detector.py # Wrapper MediaPipe
│   ├── gesture_engine.py# Reconnaissance de gestes
│   ├── smoothing.py     # Filtre EMA
│   └── safety.py        # Guards sécurité
├── effects/
│   ├── drawing.py       # Effet dessin
│   ├── overlay.py       # Effet overlay aura
│   └── skeleton.py      # Debug squelette
├── utils/
│   └── fps_counter.py   # Compteur FPS
└── assets/
    └── aura.png         # Image overlay
```

## Roadmap

- [X] Phase 1 : PoC local (OpenCV + MediaPipe)
- [ ] Phase 2 : Conteneurisation Docker
- [ ] Phase 3 : Déploiement Edge (Raspberry Pi)
- [ ] Phase 4 : Interface Web (WebSockets)
