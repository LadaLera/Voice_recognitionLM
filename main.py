
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import pygame
import sys
import speech_recognition as sr
import random
import time
import threading
import queue
from typing import Tuple, Optional

try:
    from accent_detector import detect_accent
    ACCENT_DETECTOR_AVAILABLE = True
    print("✓ Accent detector loaded successfully")
except ImportError as e:
    print(f"⚠ Warning: accent_detector_claude not available - {e}")
    print("  Accent detection will be disabled in game")
    ACCENT_DETECTOR_AVAILABLE = False

    def detect_accent(audio):
        return "Detector not available"

try:
    from translator_mode_claude import run_translator_mode
    TRANSLATOR_AVAILABLE = True
    print("✓ Translator mode loaded")
except ImportError as e:
    print(f"⚠ Translator not available: {e}")
    TRANSLATOR_AVAILABLE = False

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("English Pronunciation Game")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 128, 255)
ORANGE = (255, 165, 0)
LIGHT_GRAY = (200, 200, 200)

font = pygame.font.Font(None, 48)
small_font = pygame.font.Font(None, 32)
tiny_font = pygame.font.Font(None, 24)

WORDS = [ "cat", "dog", "lion", "tiger", "elephant", "giraffe", "monkey", "bear", "fox", "wolf", "rabbit", "mouse", "bird", "fish", "snake", "frog", "tree", "flower", "river", "mountain", "ocean", "lake", "sky", "cloud", "sun", "moon", "star", "rain", "snow", "wind", "forest", "desert", "happy", "sad", "angry", "excited", "scared", "calm", "proud", "brave", "kind", "tired", "bored", "nervous", "worried", "surprised", "confused", "apple", "banana", "orange", "bread", "cheese", "milk", "water", "juice", "pizza", "burger", "chicken", "fish", "rice", "cake", "chocolate", "chair", "table", "bed", "lamp", "door", "window", "mirror", "carpet", "run", "walk", "jump", "sit", "stand", "talk", "listen", "read", "write", "sing", "dance", "eat", "drink", "sleep", "play", "think", "swim", "car", "bus", "train", "plane", "bike", "boat", "taxi", "truck", "morning", "afternoon", "evening", "night", "today", "tomorrow", "yesterday"]


class SpeechRecognizer:
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.is_listening = False
        self._calibrated = False
        self.result_queue = queue.Queue()
        self.is_processing = False
    
    def _recognize_thread(self):
        try:
            with sr.Microphone() as source:
                if not self._calibrated:
                    print("Calibrating...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                    self._calibrated = True
                    print(f"Threshold: {self.recognizer.energy_threshold}")
                
                print("Listening...")
                self.is_listening = True
                
                audio = self.recognizer.listen(
                    source, 
                    timeout=10,
                    phrase_time_limit=8
                )
                
                self.is_listening = False
                self.is_processing = True
                print("Processing...")
                
                try:
                    text = self.recognizer.recognize_google(audio, language='en-US').lower()
                    print(f"✓ Recognized: '{text}'")
                    
                    if ACCENT_DETECTOR_AVAILABLE:
                        print("Detecting accent...")
                        accent = detect_accent(audio)
                        print(f"✓ Accent: {accent}")
                    else:
                        accent = "Unknown"
                    
                    self.result_queue.put((True, text, accent))
                    
                except sr.UnknownValueError:
                    print("❌ Not understood")
                    self.result_queue.put((False, "not recognized", "Unknown"))
                except sr.RequestError as e:
                    print(f"❌ API error: {e}")
                    self.result_queue.put((False, "service error", "Unknown"))
                    
        except sr.WaitTimeoutError:
            print("❌ Timeout")
            self.result_queue.put((False, "timeout", "Unknown"))
        except Exception as e:
            print(f"❌ Error: {e}")
            self.result_queue.put((False, "microphone error", "Unknown"))
        finally:
            self.is_listening = False
            self.is_processing = False
    
    def start_recognition(self):
        if not self.is_listening and not self.is_processing:
            thread = threading.Thread(target=self._recognize_thread, daemon=True)
            thread.start()
    
    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def is_busy(self):
        return self.is_listening or self.is_processing


class GameState:
    
    def __init__(self):
        self.current_word = random.choice(WORDS)
        self.hearts_left = 3
        self.total_words = 0
        self.errors = 0
        self.start_time = time.time()
        self.message = "Press Record or Space to start!"
        self.background_color = WHITE
        self.game_over = False
        self.won = False
        self.last_accent = None
        self.is_recording = False
    
    def reset(self):
        self.__init__()
    
    def next_word(self):
        old_word = self.current_word
        while self.current_word == old_word:
            self.current_word = random.choice(WORDS)
        self.total_words += 1
    
    def mark_correct(self):
        self.message = "Correct! ✓"
        self.background_color = GREEN
        self.next_word()
    
    def mark_wrong(self, recognized: str, accent: str):
        self.last_accent = accent
        
        if recognized == "not recognized":
            self.message = "Could not hear you - try again!"
        elif recognized == "timeout":
            self.message = "No speech detected - speak louder!"
        elif recognized in ["service error", "microphone error"]:
            self.message = f"Error: {recognized}"
        else:
            self.message = f"Wrong: '{recognized}'"
        
        self.background_color = RED
        self.hearts_left -= 1
        self.errors += 1
        self.total_words += 1
        
        if self.hearts_left <= 0:
            self.game_over = True
    
    def skip_word(self):
        self.message = "Word skipped"
        self.background_color = ORANGE
        self.current_word = random.choice(WORDS)
    
    def check_win_condition(self):
        if self.total_words >= 10:
            self.game_over = True
            self.won = True
    
    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    def get_score(self) -> int:
        return self.total_words - self.errors


def draw_heart(x: int, y: int):
    pygame.draw.circle(screen, BLACK, (x + 10, y), 8, 2)
    pygame.draw.circle(screen, RED, (x + 10, y), 7)
    pygame.draw.circle(screen, BLACK, (x + 22, y), 8, 2)
    pygame.draw.circle(screen, RED, (x + 22, y), 7)
    
    points = [(x + 7, y + 4), (x + 25, y + 4), (x + 16, y + 18)]
    pygame.draw.polygon(screen, BLACK, points, 2)
    pygame.draw.polygon(screen, RED, points)


def draw_button(rect: pygame.Rect, text: str, color: tuple, text_color: tuple = WHITE, 
                border_color: tuple = BLACK):
    shadow_rect = rect.copy()
    shadow_rect.x += 3
    shadow_rect.y += 3
    pygame.draw.rect(screen, (100, 100, 100), shadow_rect, border_radius=10)

    pygame.draw.rect(screen, color, rect, border_radius=10)
    pygame.draw.rect(screen, border_color, rect, 3, border_radius=10)

    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)


def draw_game_screen(state):
    screen.fill(state.background_color)

    word_text = font.render(state.current_word.upper(), True, BLACK)
    word_rect = word_text.get_rect(center=(WIDTH // 2, HEIGHT // 3))

    padding = 20
    word_bg_rect = pygame.Rect(
        word_rect.x - padding,
        word_rect.y - padding,
        word_rect.width + padding * 2,
        word_rect.height + padding * 2
    )
    pygame.draw.rect(screen, WHITE, word_bg_rect, border_radius=15)
    pygame.draw.rect(screen, BLACK, word_bg_rect, 3, border_radius=15)
    screen.blit(word_text, word_rect)

    message_y = HEIGHT // 2 + 40
    message_text = small_font.render(state.message, True, BLACK)
    screen.blit(message_text, (WIDTH // 2 - message_text.get_width() // 2, message_y))

    if state.last_accent and state.last_accent not in ["Unknown", "Detector not available"]:
        accent_y = message_y + 40  
        accent_text = tiny_font.render(f"Detected accent: {state.last_accent}", True, (0, 0, 128))
        screen.blit(accent_text, (WIDTH // 2 - accent_text.get_width() // 2, accent_y))

    if state.is_recording:
        pulse_size = 10 + abs(int(time.time() * 10) % 10 - 5)
        pygame.draw.circle(screen, RED, (WIDTH // 2, HEIGHT // 3 - 80), pulse_size)
        pygame.draw.circle(screen, (255, 100, 100), (WIDTH // 2, HEIGHT // 3 - 80), pulse_size - 2)
        
        recording_text = small_font.render("🎙️ RECORDING - SPEAK NOW!", True, RED)
        screen.blit(recording_text, (WIDTH // 2 - recording_text.get_width() // 2, HEIGHT // 3 - 120))

    hearts_text = small_font.render("Lives:", True, BLACK)
    screen.blit(hearts_text, (20, 15))
    for i in range(state.hearts_left):
        draw_heart(100 + i * 40, 20)

    progress_text = small_font.render(f"Words: {state.total_words}/10", True, BLACK)
    screen.blit(progress_text, (WIDTH - progress_text.get_width() - 20, 20))

    score_text = small_font.render(f"Score: {state.get_score()}", True, BLACK)
    screen.blit(score_text, (WIDTH - score_text.get_width() - 20, 55))

    elapsed = int(state.get_elapsed_time())
    time_text = tiny_font.render(f"Time: {elapsed}s", True, BLACK)
    screen.blit(time_text, (20, HEIGHT - 35))

    button_record = pygame.Rect(WIDTH // 2 - 160, HEIGHT - 100, 140, 60)
    button_skip = pygame.Rect(WIDTH // 2 + 20, HEIGHT - 100, 140, 60)
    
    draw_button(button_record, "Record", BLUE)
    draw_button(button_skip, "Skip", ORANGE)

    hint_text = tiny_font.render("Press SPACE to record, S to skip", True, BLACK)
    screen.blit(hint_text, (WIDTH // 2 - hint_text.get_width() // 2, HEIGHT - 35))
    
    pygame.display.flip()
    
    return button_record, button_skip


def draw_end_screen(state: GameState) -> Tuple[pygame.Rect, pygame.Rect]:
    if state.won:
        bg_color = GREEN
        message = "🎉 Excellent Work! 🎉"
        emoji = "😊"
    else:
        bg_color = RED
        message = "💔 Game Over 💔"
        emoji = "😢"
    
    screen.fill(bg_color)

    title_text = font.render(message, True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 4))

    emoji_text = font.render(emoji, True, BLACK)
    screen.blit(emoji_text, (WIDTH // 2 - emoji_text.get_width() // 2, HEIGHT // 4 + 60))

    score = state.get_score()
    result_text = font.render(f"Score: {score}/10", True, BLACK)
    screen.blit(result_text, (WIDTH // 2 - result_text.get_width() // 2, HEIGHT // 2 - 30))

    stats_y = HEIGHT // 2 + 20
    stats = [
        f"Correct: {score}",
        f"Wrong: {state.errors}",
        f"Time: {int(state.get_elapsed_time())}s"
    ]
    for i, stat in enumerate(stats):
        stat_text = small_font.render(stat, True, BLACK)
        screen.blit(stat_text, (WIDTH // 2 - stat_text.get_width() // 2, stats_y + i * 30))

    button_retry = pygame.Rect(WIDTH // 2 - 220, HEIGHT - 120, 180, 60)
    button_menu = pygame.Rect(WIDTH // 2 + 40, HEIGHT - 120, 180, 60)
    
    draw_button(button_retry, "Retry", BLUE)
    draw_button(button_menu, "Menu", GREEN)
    
    pygame.display.flip()
    
    return button_retry, button_menu


def main_game():
    state = GameState()
    recognizer = SpeechRecognizer()
    
    clock = pygame.time.Clock()
    running = True
    waiting_for_result = False
    
    while running:
        clock.tick(60)
        if state.game_over:
            button_retry, button_menu = draw_end_screen(state)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if button_retry.collidepoint(event.pos):
                        state.reset()
                        recognizer = SpeechRecognizer()
                        waiting_for_result = False
                    elif button_menu.collidepoint(event.pos):
                        return "menu"
            continue

        state.is_recording = recognizer.is_busy()

        if waiting_for_result:
            result = recognizer.get_result()
            if result:
                success, recognized_text, accent = result
                waiting_for_result = False
                
                if success and recognized_text == state.current_word:
                    state.mark_correct()
                else:
                    state.mark_wrong(recognized_text, accent)
                
                state.check_win_condition()
        
        button_record, button_skip = draw_game_screen(state)

        if state.get_elapsed_time() > 300:
            state.game_over = True
            state.won = False
            continue
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_record.collidepoint(event.pos) and not recognizer.is_busy():
                    recognizer.start_recognition()
                    waiting_for_result = True
                
                elif button_skip.collidepoint(event.pos):
                    state.skip_word()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not recognizer.is_busy():
                    recognizer.start_recognition()
                    waiting_for_result = True
                
                elif event.key == pygame.K_s:
                    state.skip_word()
                
                elif event.key == pygame.K_ESCAPE:
                    return "menu"


def detect_accent_mode():
    recognizer = SpeechRecognizer()
    result_text = ""
    waiting_for_result = False
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        clock.tick(60)

        if waiting_for_result:
            result = recognizer.get_result()
            if result:
                _, _, accent = result
                result_text = f"Detected: {accent}"
                waiting_for_result = False
        
        screen.fill(WHITE)

        title = font.render("🎤 Detect Your Accent", True, BLACK)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 5))

        instruction = small_font.render("Say something in English", True, BLACK)
        screen.blit(instruction, (WIDTH // 2 - instruction.get_width() // 2, HEIGHT // 5 + 60))

        if recognizer.is_busy():
            pulse_size = 15 + abs(int(time.time() * 10) % 10 - 5)
            pygame.draw.circle(screen, RED, (WIDTH // 2, HEIGHT // 2 - 60), pulse_size)
            
            status = "LISTENING..." if recognizer.is_listening else "PROCESSING..."
            recording_text = small_font.render(status, True, RED)
            screen.blit(recording_text, (WIDTH // 2 - recording_text.get_width() // 2, HEIGHT // 2 - 20))

        if result_text:
            result = small_font.render(result_text, True, BLACK)
            screen.blit(result, (WIDTH // 2 - result.get_width() // 2, HEIGHT // 2 + 20))

        button_record = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 180, 200, 60)
        button_back = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 100, 200, 60)
        
        draw_button(button_record, "Record", BLUE)
        draw_button(button_back, "Back", GREEN)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_record.collidepoint(event.pos) and not recognizer.is_busy():
                    result_text = ""
                    recognizer.start_recognition()
                    waiting_for_result = True
                
                elif button_back.collidepoint(event.pos):
                    return "menu"
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "menu"


def main_menu():
    clock = pygame.time.Clock()
    running = True
    
    while running:
        clock.tick(60)
        screen.fill(WHITE)

        title1 = font.render("English Pronunciation", True, BLACK)
        title2 = font.render("Trainer", True, BLUE)
        screen.blit(title1, (WIDTH // 2 - title1.get_width() // 2, HEIGHT // 5 - 20))
        screen.blit(title2, (WIDTH // 2 - title2.get_width() // 2, HEIGHT // 5 + 35))

        if ACCENT_DETECTOR_AVAILABLE:
            status_text = tiny_font.render("✓ Accent detector: Ready", True, GREEN)
        else:
            status_text = tiny_font.render("⚠ Accent detector: Not available", True, ORANGE)
        screen.blit(status_text, (WIDTH // 2 - status_text.get_width() // 2, HEIGHT // 5 + 85))

        instruction = tiny_font.render("💡 TIP: Speak CLEARLY and LOUDER than normal!", True, BLUE)
        screen.blit(instruction, (WIDTH // 2 - instruction.get_width() // 2, HEIGHT // 5 + 110))
 
        button_start_y = HEIGHT // 2 + 20 
        button_spacing = 70
        
        button_game = pygame.Rect(WIDTH // 2 - 150, button_start_y, 300, 60)
        button_accent = pygame.Rect(WIDTH // 2 - 150, button_start_y + button_spacing, 300, 60)
        button_translator = pygame.Rect(WIDTH // 2 - 150, button_start_y + button_spacing * 2, 300, 60)
        button_exit = pygame.Rect(WIDTH // 2 - 150, button_start_y + button_spacing * 3, 300, 60)
        
        draw_button(button_game, "Start Game", BLUE)
        draw_button(button_accent, "Detect Accent", GREEN)
        draw_button(button_translator, "Translator", ORANGE)
        draw_button(button_exit, "Exit", RED)

        hint_text = tiny_font.render("Press ESC to return to menu", True, BLACK)
        screen.blit(hint_text, (WIDTH // 2 - hint_text.get_width() // 2, HEIGHT - 30))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_game.collidepoint(event.pos):
                    result = main_game()
                    if result == "quit":
                        return
                
                elif button_accent.collidepoint(event.pos):
                    result = detect_accent_mode()
                    if result == "quit":
                        return
                
                elif button_translator.collidepoint(event.pos):
                    if TRANSLATOR_AVAILABLE:
                        result = run_translator_mode(screen, font, small_font, tiny_font)
                        if result == "quit":
                            return
                    else:
                        print("⚠ Translator mode not available!")
                        print("   Make sure translator_mode.py exists")
                
                elif button_exit.collidepoint(event.pos):
                    return


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("🎮 English Pronunciation Trainer")
        print("=" * 60)
        print()
        
        if ACCENT_DETECTOR_AVAILABLE:
            print("✓ Accent detector loaded")
        else:
            print("⚠ Accent detector not available (limited functionality)")
        
        print()
        print("Starting game...")
        print("=" * 60)
        
        main_menu()
        
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        sys.exit()