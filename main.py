"""
CatchMyHands — Main Entry Point
===================================
Boucle principale du système de vision interactive.
Capture vidéo → Détection IA → Gestes → Effets visuels → Affichage.

Contrôles clavier :
    Q / ESC   : Quitter
    D         : Toggle squelette
    I         : Toggle IDs des landmarks
    S         : Screenshot
    +/-       : Ajuster seuil de pincement
    1 / 2     : Activer/Désactiver les filtres (1: Scanner B&W, 2: Pixelate)
"""

import os
import sys
import time
import glob

# Prévenir les plantages d'encodage sur Windows (ex. console GBK/CP936)
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

import cv2
import numpy as np

import config
from core.hand_detector import HandDetector
from core.gesture_engine import GestureEngine, GestureType
from core.smoothing import LandmarkSmoother
from core.safety import SafetyGuard
from effects.overlay import OverlayEffect
from effects.skeleton import SkeletonRenderer
from effects.box_frame import BoxFrameEffect
from effects.menu_hud import MenuHUDEffect
from effects.finger_filter import FingerFilterEffect
from effects.main_menu import MainMenuRenderer
from effects.games_menu import GamesMenuRenderer
from core.snake_game import SnakeGame
from core.pong_game import PongGame
from utils.fps_counter import FPSCounter


def list_available_cameras():
    """Détecte et liste les caméras disponibles sur la machine."""
    cameras = []
    if sys.platform.startswith('linux'):
        for path in sorted(glob.glob('/sys/class/video4linux/video*')):
            try:
                device_num = int(path.split('/')[-1].replace('video', ''))
                with open(os.path.join(path, 'name'), 'r') as f:
                    name = f.read().strip()
                dev_path = f"/dev/video{device_num}"
                # Test d'ouverture rapide
                cap = cv2.VideoCapture(dev_path)
                if cap.isOpened():
                    cameras.append((device_num, dev_path, name, True))
                    cap.release()
                else:
                    cameras.append((device_num, dev_path, name, False))
            except Exception:
                pass
    else:
        # Windows / macOS fallback
        # Scanner uniquement les 3 premiers index (0, 1, 2) pour accélérer le démarrage.
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append((i, f"Index {i}", "Caméra générique", True))
                cap.release()
    return cameras


def get_best_camera_index():
    """
    Détermine l'index de la meilleure caméra disponible.
    Préfère une caméra externe branchée (index != 0) si disponible,
    sinon se rabat sur la caméra interne (index 0).
    """
    cams = list_available_cameras()
    valid_cams = [c for c in cams if c[3]]  # Garder uniquement celles avec OK=True
    
    if not valid_cams:
        print("⚠ Aucune caméra fonctionnelle détectée sur le système, repli sur l'index 0.")
        return 0
        
    # Chercher les caméras externes (index différent de 0)
    external_cams = [c for c in valid_cams if c[0] != 0]
    if external_cams:
        best_cam = external_cams[0]
        print(f"[Auto-Camera] Caméra externe détectée : Index {best_cam[0]} ({best_cam[1]}) - {best_cam[2]}")
        return best_cam[0]
        
    # Sinon utiliser la caméra par défaut (index 0)
    best_cam = valid_cams[0]
    print(f"[Auto-Camera] Caméra interne par défaut : Index {best_cam[0]} ({best_cam[1]}) - {best_cam[2]}")
    return best_cam[0]


class CatchMyHands:
    """
    Application principale CatchMyHands.

    Orchestre tous les modules : détection, gestes, effets, affichage.
    """

    def __init__(self):
        """Initialise tous les composants du pipeline."""
        print("=" * 50)
        print("  🖐️  CatchMyHands — Phase 1 PoC")
        print("=" * 50)

        # ── Pipeline IA ──
        self.detector = HandDetector()
        self.gesture_engine = GestureEngine()
        self.smoother = LandmarkSmoother()
        self.safety = SafetyGuard()

        # ── Effets visuels ──
        self.overlay = OverlayEffect()
        self.skeleton = SkeletonRenderer()
        self.box_frame = BoxFrameEffect()
        self.menu_hud = MenuHUDEffect()
        self.finger_filter = FingerFilterEffect()

        # ── État ──
        self.show_skeleton = config.SKELETON_ENABLED_DEFAULT
        self.fps_counter = FPSCounter()
        self.frame_count = 0
        self.start_time = None
        self._last_frame = None  # Dernière frame pour le screenshot
        self._smoothed_hands_this_frame = []
        self._hands_metadata = []
        self.is_two_hand_frame_active = False
        # Filtres activés dans l'ordre d'activation (ex: ["bw", "pixelate"])
        self.active_filters_ordered = []

        # ── Composants Menu et Jeux ──
        self.main_menu = MainMenuRenderer()
        self.games_menu = GamesMenuRenderer()
        self.snake_game = SnakeGame()
        self.pong_game = PongGame()
        self.state = "MENU"  # "MENU", "FILTERS", "GAMES_MENU", "GAME_SNAKE", "GAME_PONG"
        self.transition_cooldown = 0
        self.width = config.CAMERA_WIDTH
        self.height = config.CAMERA_HEIGHT

    def run(self):
        """Lance la boucle principale."""
        # ── Ouvrir la caméra ──
        camera_id = config.CAMERA_INDEX
        if camera_id == "auto":
            camera_id = get_best_camera_index()
        elif isinstance(camera_id, str) and camera_id.isdigit():
            camera_id = int(camera_id)

        # Utiliser cv2.CAP_DSHOW sur Windows pour forcer le support des hauts framerates (ex: 60 FPS)
        if sys.platform.startswith('win'):
            cap = cv2.VideoCapture(camera_id if isinstance(camera_id, int) else 0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_id)

        # Forcer le codec MJPEG pour un framerate optimal (évite le goulot d'étranglement USB)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la caméra !")
            print(f"   Identifiant testé : {config.CAMERA_INDEX}")
            print("\n🔍 Détection des caméras disponibles sur le système :")
            cams = list_available_cameras()
            if cams:
                for num, path, name, ok in cams:
                    status = "Disponible ✅" if ok else "Indisponible (Flux occupé ou métadonnées seules) ❌"
                    print(f"   - Index {num} ({path}) : {name} | Statut : {status}")
                print("\n💡 Astuce : Modifiez CAMERA_INDEX dans config.py avec l'index ou le chemin souhaité.")
            else:
                print("   Aucune caméra détectée sur le système.")
            sys.exit(1)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = actual_w
        self.height = actual_h
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[Camera] Ouverte — {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")



        # ── Fenêtre plein écran ──
        cv2.namedWindow("CatchMyHands", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("CatchMyHands", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.start_time = time.time()
        print("\n🎮 Contrôles :")
        print("   Q/ESC  Quitter | D  Squelette | I  IDs landmarks")
        print("   S  Screenshot | +/-  Seuil pincement")
        print("-" * 50)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠ Frame perdu, nouvelle tentative...")
                    continue

                if self.transition_cooldown > 0:
                    self.transition_cooldown -= 1

                # ── Miroir horizontal (plus naturel pour l'utilisateur) ──
                frame = cv2.flip(frame, 1)

                # ── Envoyer au détecteur IA ──
                timestamp_ms = int(time.time() * 1000)
                self.detector.detect_async(frame, timestamp_ms)

                # ── Collecter les données de chaque main détectée ──
                num_hands = self.detector.get_num_hands_detected()
                self._smoothed_hands_this_frame = []
                self._hands_metadata = []

                for hand_idx in range(num_hands):
                    self._collect_hand_data(hand_idx)
                # ── Rendu selon l'état actuel ──
                if self.state == "MENU":
                    # Squelettes en arrière-plan si activé
                    if self.show_skeleton:
                        for hand_data in self._hands_metadata:
                            frame = self.skeleton.render(frame, hand_data["smoothed"], hand_data["handedness"], hand_data["edge_factor"])
                    
                    frame, menu_action = self.main_menu.render(frame, self._hands_metadata, block_clicks=(self.transition_cooldown > 0))
                    if menu_action == 1:
                        self.set_state("FILTERS")
                    elif menu_action == 2:
                        self.set_state("GAMES_MENU")

                elif self.state == "GAMES_MENU":
                    # Squelettes en arrière-plan si activé
                    if self.show_skeleton:
                        for hand_data in self._hands_metadata:
                            frame = self.skeleton.render(frame, hand_data["smoothed"], hand_data["handedness"], hand_data["edge_factor"])
                    
                    frame, games_menu_action = self.games_menu.render(frame, self._hands_metadata, block_clicks=(self.transition_cooldown > 0))
                    if games_menu_action == 1:
                        self.set_state("GAME_SNAKE")
                        self.snake_game.reset(self.width, self.height)
                    elif games_menu_action == 2:
                        self.set_state("GAME_PONG")
                        self.pong_game.reset(self.width, self.height)

                elif self.state == "FILTERS":
                    # ── Rendre tous les effets de cadres (bi-manuel ou solo) ──
                    frame = self._render_frame_effects(frame)

                    # ── HUD (Heads-Up Display) ──
                    frame = self._render_hud(frame, num_hands)

                    # ── Menu Latéral (Options) ──
                    frame = self.menu_hud.render(
                        frame,
                        self.active_filters_ordered
                    )

                elif self.state == "GAME_SNAKE":
                    # Mode Jeux (Snake)
                    frame = self.snake_game.update_and_render(frame, self._hands_metadata)

                elif self.state == "GAME_PONG":
                    # Mode Jeux (Pong)
                    frame = self.pong_game.update_and_render(frame, self._hands_metadata)

                # ── Affichage ──
                cv2.imshow("CatchMyHands", frame)
                self._last_frame = frame  # Sauvegarder pour screenshot
                self.fps_counter.tick()
                self.frame_count += 1

                # ── Gestion des touches ──
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break

        except KeyboardInterrupt:
            print("\n⏹ Interrompu par l'utilisateur")
        except BaseException as e:
            import traceback
            print(f"\n❌ Erreur inattendue : {type(e).__name__} - {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            # ── Nettoyage ──
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()
            self.box_frame.close()

            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"\n📊 Stats : {self.frame_count} frames en {elapsed:.1f}s "
                  f"(moyenne {avg_fps:.1f} FPS)")

    def _collect_hand_data(self, hand_idx: int):
        """
        Collecte, lisse et valide les landmarks d'une main.
        Stocke les landmarks et métadonnées si valides.
        """
        raw_landmarks = self.detector.get_landmarks_array(hand_idx)
        if raw_landmarks is None:
            return

        smoothed = self.smoother.smooth(raw_landmarks, hand_idx)
        if smoothed is None:
            return

        safety_report = self.safety.validate(smoothed, hand_idx)
        if not safety_report["valid"]:
            return

        self._smoothed_hands_this_frame.append(smoothed)

        edge_factor = safety_report["edge_factor"]
        gesture = self.gesture_engine.analyze(smoothed, hand_idx)
        handedness = self.detector.get_handedness(hand_idx)

        self._hands_metadata.append({
            "smoothed": smoothed,
            "gesture": gesture,
            "handedness": handedness,
            "edge_factor": edge_factor
        })

    def _render_frame_effects(self, frame: np.ndarray) -> np.ndarray:
        """
        Orchestre le rendu des cadres bi-manuels (2 mains valides)
        ou des mini-cadres mono-main (1 main valide).
        """
        num_valid_hands = len(self._smoothed_hands_this_frame)

        if num_valid_hands == 2:
            self.is_two_hand_frame_active = True  # Indique qu'on est en mode bi-manuel
            smoothed0 = self._smoothed_hands_this_frame[0]
            smoothed1 = self._smoothed_hands_this_frame[1]

            # Déterminer main gauche et droite sur l'écran
            if smoothed0[0][0] < smoothed1[0][0]:
                lm_left, lm_right = smoothed0, smoothed1
            else:
                lm_left, lm_right = smoothed1, smoothed0

            # 1. Rendre les 4 cadres bi-manuels
            if self.active_filters_ordered:
                frame = self.finger_filter.render_two_hands(
                    frame, lm_left, lm_right,
                    self.active_filters_ordered
                )

            # 2. Rendre les auras et squelettes individuels (sans filtre de doigt mono-main)
            for hand_data in self._hands_metadata:
                frame = self._render_individual_effects(frame, hand_data)

            # 3. Rendre l'ancien cadre L-shape s'il est actif
            is_l_shape_active = self.gesture_engine.check_two_hand_frame(lm_left, lm_right)
            if is_l_shape_active:
                bw = "bw" in self.active_filters_ordered
                pix = "pixelate" in self.active_filters_ordered
                frame = self.box_frame.render(frame, lm_left, lm_right, bw_filter=bw, pix_filter=pix)

        elif num_valid_hands == 1:
            self.is_two_hand_frame_active = False
            hand_data = self._hands_metadata[0]

            # 1. Rendre les mini-cadres de filtres mono-main
            if self.active_filters_ordered:
                frame = self.finger_filter.render_single_hand(
                    frame, hand_data["smoothed"],
                    self.active_filters_ordered
                )

            # 2. Rendre l'aura et le squelette de cette main
            frame = self._render_individual_effects(frame, hand_data)
        else:
            self.is_two_hand_frame_active = False

        return frame

    def _render_individual_effects(self, frame: np.ndarray, hand_data: dict) -> np.ndarray:
        """
        Dessine l'aura et le squelette d'une main.
        """
        smoothed = hand_data["smoothed"]
        gesture = hand_data["gesture"]
        handedness = hand_data["handedness"]
        edge_factor = hand_data["edge_factor"]

        # 1. Effet aura (main ouverte)
        if gesture.gesture_type == GestureType.OPEN_HAND and gesture.palm_center:
            frame = self.overlay.render(
                frame,
                gesture.palm_center,
                gesture.hand_size,
                edge_factor
            )

        # 2. Squelette (debug)
        if self.show_skeleton:
            frame = self.skeleton.render(
                frame, smoothed, handedness, edge_factor
            )

        return frame

    def _render_hud(self, frame: np.ndarray, num_hands: int) -> np.ndarray:
        """Affiche le HUD (FPS, état, geste actif)."""
        h, w = frame.shape[:2]

        # ── Fond semi-transparent pour le HUD ──
        hud_h = 35
        overlay = frame[:hud_h, :].copy()
        cv2.rectangle(frame, (0, 0), (w, hud_h), config.HUD_BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.3, frame[:hud_h, :], 0.7, 0, frame[:hud_h, :])

        # ── FPS ──
        fps_text = self.fps_counter.fps_str
        cv2.putText(frame, fps_text, config.HUD_POSITION,
                    cv2.FONT_HERSHEY_SIMPLEX, config.HUD_FONT_SCALE,
                    (0, 255, 128), 2, cv2.LINE_AA)

        # ── Nombre de mains ──
        hands_text = f"Mains: {num_hands}"
        cv2.putText(frame, hands_text, (120, config.HUD_POSITION[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, config.HUD_FONT_SCALE,
                    config.HUD_COLOR, 1, cv2.LINE_AA)

        # ── Geste actif ──
        result = self.detector.get_result()
        if result and len(result.hand_landmarks) > 0:
            landmarks = self.detector.get_landmarks_array(0)
            if landmarks is not None:
                gesture = self.gesture_engine.analyze(landmarks, hand_index=99)
                gesture_labels = {
                    GestureType.PINCH: "✏️ PINCH",
                    GestureType.OPEN_HAND: "🖐️ OPEN",
                    GestureType.FIST: "✊ FIST",
                    GestureType.NONE: "—",
                }
                gesture_text = gesture_labels.get(gesture.gesture_type, "—")
                cv2.putText(frame, gesture_text, (270, config.HUD_POSITION[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, config.HUD_FONT_SCALE,
                            (0, 200, 255), 1, cv2.LINE_AA)

        # ── Mode squelette ──
        if self.show_skeleton:
            cv2.putText(frame, "[SKEL]", (w - 80, config.HUD_POSITION[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (100, 100, 255), 1, cv2.LINE_AA)

        return frame

    def _handle_key(self, key: int) -> bool:
        """
        Gère les entrées clavier.

        Returns:
            False pour quitter, True sinon.
        """
        if key == ord('q') or key == 27:  # Q ou ESC
            if self.state == "MENU":
                print("👋 Au revoir !")
                return False
            elif self.state in ["GAMES_MENU", "FILTERS"]:
                self.set_state("MENU")
                return True
            elif self.state in ["GAME_SNAKE", "GAME_PONG"]:
                self.set_state("GAMES_MENU")
                return True

        elif key == ord('m') or key == ord('M'):
            if self.state in ["GAMES_MENU", "FILTERS"]:
                self.set_state("MENU")
                return True
            elif self.state in ["GAME_SNAKE", "GAME_PONG"]:
                self.set_state("GAMES_MENU")
                return True

        elif key == ord('d'):
            self.show_skeleton = not self.show_skeleton
            print(f"[Mode] Squelette {'activé' if self.show_skeleton else 'désactivé'}")

        elif key == ord('i'):
            self.skeleton.toggle_ids()

        elif key == ord('1') or key == ord('&'):
            if self.state == "FILTERS":
                self._toggle_filter("bw")
            elif self.state == "MENU":
                self.set_state("FILTERS")
            elif self.state == "GAMES_MENU":
                self.set_state("GAME_SNAKE")
                self.snake_game.reset(self.width, self.height)

        elif key == ord('2') or key == ord('é'):
            if self.state == "FILTERS":
                self._toggle_filter("pixelate")
            elif self.state == "MENU":
                self.set_state("GAMES_MENU")
            elif self.state == "GAMES_MENU":
                self.set_state("GAME_PONG")
                self.pong_game.reset(self.width, self.height)

        elif key == ord('3') or key == ord('"'):
            if self.state == "FILTERS":
                self._toggle_filter("invert")

        elif key == ord('4') or key == ord('\''):
            if self.state == "FILTERS":
                self._toggle_filter("edge")

        elif key == ord('r') or key == ord('R'):
            if self.state == "GAME_SNAKE":
                self.snake_game.reset(self.width, self.height)
                print("[Jeu] Snake réinitialisé")
            elif self.state == "GAME_PONG":
                self.pong_game.reset(self.width, self.height)
                print("[Jeu] Pong réinitialisé")

        elif key == ord('s'):
            self._take_screenshot()

        elif key == ord('+') or key == ord('='):
            config.PINCH_THRESHOLD = min(0.15, config.PINCH_THRESHOLD + 0.005)
            config.PINCH_RELEASE_THRESHOLD = config.PINCH_THRESHOLD + 0.02
            print(f"[Config] Seuil pincement → {config.PINCH_THRESHOLD:.3f}")

        elif key == ord('-'):
            config.PINCH_THRESHOLD = max(0.01, config.PINCH_THRESHOLD - 0.005)
            config.PINCH_RELEASE_THRESHOLD = config.PINCH_THRESHOLD + 0.02
            print(f"[Config] Seuil pincement → {config.PINCH_THRESHOLD:.3f}")

        return True

    def _toggle_filter(self, filter_name: str):
        """
        Active ou désactive un filtre et met à jour la liste ordonnée.
        L'ordre dans la liste détermine l'attribution aux doigts.
        """
        display_name = config.FILTER_DISPLAY_NAMES.get(filter_name, filter_name)

        if filter_name in self.active_filters_ordered:
            self.active_filters_ordered.remove(filter_name)
            print(f"[Filtre] {display_name} désactivé")
        else:
            if len(self.active_filters_ordered) >= 4:
                print(f"[Filtre] Maximum 4 filtres actifs ! Désactivez-en un d'abord.")
                return
            self.active_filters_ordered.append(filter_name)
            slot_idx = len(self.active_filters_ordered) - 1
            finger_name = config.FINGER_DISPLAY_NAMES[slot_idx]
            print(f"[Filtre] {display_name} activé → Pouce ↔ {finger_name}")

        # Log de l'état actuel
        if self.active_filters_ordered:
            mapping = []
            for i, f in enumerate(self.active_filters_ordered):
                fname = config.FINGER_DISPLAY_NAMES[i]
                dname = config.FILTER_DISPLAY_NAMES.get(f, f)
                mapping.append(f"{fname}={dname}")
            print(f"[Filtres] Attribution : {' | '.join(mapping)}")
        else:
            print("[Filtres] Aucun filtre actif")

    def _take_screenshot(self):
        """Sauvegarde le frame actuel en PNG dans le dossier screenshots/."""
        if self._last_frame is None:
            print("[Screenshot] Aucune frame disponible.")
            return
        os.makedirs("screenshots", exist_ok=True)
        filename = f"screenshots/catch_{int(time.time())}.png"
        success = cv2.imwrite(filename, self._last_frame)
        if success:
            print(f"[Screenshot] ✅ Sauvegardé → {filename}")
        else:
            print(f"[Screenshot] ❌ Échec de la sauvegarde.")

    def set_state(self, new_state: str):
        """Met à jour l'état de l'application et réinitialise le cooldown de clic."""
        if self.state != new_state:
            self.state = new_state
            self.transition_cooldown = 30  # Cooldown de 30 frames (~0.5s)
            print(f"[Navigation] Passage à l'état : {new_state}")


def disable_quick_edit():
    """Désactive le QuickEdit Mode dans le terminal Windows pour éviter les blocages lors de clics/frappes."""
    import sys
    if sys.platform.startswith('win'):
        try:
            kernel32 = ctypes.windll.kernel32
            # STD_INPUT_HANDLE = -10
            h_input = kernel32.GetStdHandle(-10)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(h_input, ctypes.byref(mode)):
                # ENABLE_QUICK_EDIT_LINE = 0x0040, ENABLE_EXTENDED_FLAGS = 0x0080
                # On enlève le flag QuickEdit (0x0040) et on s'assure d'avoir l'extended flag (0x0080)
                new_mode = (mode.value & ~0x0040) | 0x0080
                kernel32.SetConsoleMode(h_input, new_mode)
                print("[Windows Console] QuickEdit Mode désactivé (évite les blocages)")
        except Exception:
            pass


def main():
    """Point d'entrée."""
    disable_quick_edit()
    app = CatchMyHands()
    app.run()


if __name__ == "__main__":
    main()
