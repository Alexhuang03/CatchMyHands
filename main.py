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
from effects.glass_shatter import GlassShatterEffect
from effects.menu_hud import MenuHUDEffect
from utils.fps_counter import FPSCounter


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
        self.glass_shatter = GlassShatterEffect()
        self.menu_hud = MenuHUDEffect()

        # ── État ──
        self.show_skeleton = config.SKELETON_ENABLED_DEFAULT
        self.fps_counter = FPSCounter()
        self.frame_count = 0
        self.start_time = None
        self._last_frame = None  # Dernière frame pour le screenshot
        self._smoothed_hands_this_frame = []
        self.is_two_hand_frame_active = False
        self._was_frame_active = False  # Pour détecter la transition active → inactive
        self.option_bw_active = False
        self.option_pixelate_active = False

    def run(self):
        """Lance la boucle principale."""
        # ── Ouvrir la caméra ──
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la caméra !")
            print(f"   Index testé : {config.CAMERA_INDEX}")
            print("   Vérifiez que la webcam est connectée et accessible.")
            sys.exit(1)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

                # ── Miroir horizontal (plus naturel pour l'utilisateur) ──
                frame = cv2.flip(frame, 1)

                # ── Envoyer au détecteur IA ──
                timestamp_ms = int(time.time() * 1000)
                self.detector.detect_async(frame, timestamp_ms)

                # ── Traiter chaque main détectée ──
                num_hands = self.detector.get_num_hands_detected()
                self._smoothed_hands_this_frame = []

                for hand_idx in range(num_hands):
                    frame = self._process_hand(frame, hand_idx)

                # ── Geste et effet à deux mains ──
                if num_hands == 2:
                    frame = self._process_two_hands(frame)
                else:
                    # Détecter la disparition du cadre pour déclencher le bris de glace
                    if self._was_frame_active and (self.option_bw_active or self.option_pixelate_active):
                        if self.box_frame.last_box is not None:
                            x1, y1, x2, y2 = self.box_frame.last_box
                            self.glass_shatter.trigger(frame, x1, y1, x2, y2, bw_active=self.option_bw_active)
                            self.box_frame.last_box = None
                    self.is_two_hand_frame_active = False

                # Mettre à jour l'état précédent
                self._was_frame_active = self.is_two_hand_frame_active

                # ── Animation bris de glace ──
                if self.glass_shatter.is_animating:
                    frame = self.glass_shatter.render(frame)

                # ── HUD (Heads-Up Display) ──
                frame = self._render_hud(frame, num_hands)

                # ── Menu Latéral (Options) ──
                frame = self.menu_hud.render(
                    frame,
                    self.option_bw_active,
                    self.option_pixelate_active
                )

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

    def _process_hand(self, frame: np.ndarray, hand_idx: int) -> np.ndarray:
        """
        Pipeline complet pour une main détectée :
        landmarks → lissage → sécurité → geste → effets.
        """
        # ── Récupérer les landmarks bruts ──
        raw_landmarks = self.detector.get_landmarks_array(hand_idx)
        if raw_landmarks is None:
            return frame

        # ── Lissage ──
        smoothed = self.smoother.smooth(raw_landmarks, hand_idx)
        if smoothed is None:
            return frame

        # ── Vérification de sécurité ──
        safety_report = self.safety.validate(smoothed, hand_idx)
        if not safety_report["valid"]:
            return frame

        self._smoothed_hands_this_frame.append(smoothed)

        edge_factor = safety_report["edge_factor"]

        # ── Détection de geste ──
        gesture = self.gesture_engine.analyze(smoothed, hand_idx)
        handedness = self.detector.get_handedness(hand_idx)

        # ── Appliquer les effets ──
        # Ne pas dessiner les effets individuels (aura et squelette) si le cadre à deux mains est actif ou en cours de transition
        is_frame_rendering = self.is_two_hand_frame_active or self._was_frame_active

        if not is_frame_rendering:
            # Effet overlay aura (main ouverte)
            if gesture.gesture_type == GestureType.OPEN_HAND and gesture.palm_center:
                frame = self.overlay.render(
                    frame,
                    gesture.palm_center,
                    gesture.hand_size,
                    edge_factor
                )

            # Squelette (debug)
            if self.show_skeleton:
                frame = self.skeleton.render(
                    frame, smoothed, handedness, edge_factor
                )

        return frame

    def _process_two_hands(self, frame: np.ndarray) -> np.ndarray:
        """
        Gère le geste et l'effet à deux mains lorsque deux mains sont détectées.
        """
        if len(self._smoothed_hands_this_frame) < 2:
            return frame

        smoothed0 = self._smoothed_hands_this_frame[0]
        smoothed1 = self._smoothed_hands_this_frame[1]

        # Déterminer quelle main est à gauche et à droite sur l'écran (selon coordonnée X du poignet)
        if smoothed0[0][0] < smoothed1[0][0]:
            lm_left, lm_right = smoothed0, smoothed1
        else:
            lm_left, lm_right = smoothed1, smoothed0

        # Vérifier si le geste de cadre est actif
        was_active_before = self.is_two_hand_frame_active
        self.is_two_hand_frame_active = self.gesture_engine.check_two_hand_frame(lm_left, lm_right)

        if self.is_two_hand_frame_active:
            frame = self.box_frame.render(frame, lm_left, lm_right, bw_filter=self.option_bw_active, pix_filter=self.option_pixelate_active)
        elif was_active_before and (self.option_bw_active or self.option_pixelate_active):
            # Le cadre vient de disparaître → déclencher le bris de glace
            if self.box_frame.last_box is not None:
                x1, y1, x2, y2 = self.box_frame.last_box
                self.glass_shatter.trigger(frame, x1, y1, x2, y2, bw_active=self.option_bw_active)
                self.box_frame.last_box = None

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
            print("👋 Au revoir !")
            return False

        elif key == ord('d'):
            self.show_skeleton = not self.show_skeleton
            print(f"[Mode] Squelette {'activé' if self.show_skeleton else 'désactivé'}")

        elif key == ord('i'):
            self.skeleton.toggle_ids()

        elif key == ord('1') or key == ord('&'):
            self.option_bw_active = not self.option_bw_active
            print(f"[Option 1] Scanner B&W {'activé' if self.option_bw_active else 'désactivé'}")

        elif key == ord('2') or key == ord('é'):
            self.option_pixelate_active = not self.option_pixelate_active
            print(f"[Option 2] Pixelate {'activé' if self.option_pixelate_active else 'désactivé'}")



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


def main():
    """Point d'entrée."""
    app = CatchMyHands()
    app.run()


if __name__ == "__main__":
    main()
