# import pygame
from functools import cache
import traceback

# NORMAL = (128, 128, 128)
# RED = (255, 0, 0)
# GREEN = (0, 255, 0)

# KEY_START = pygame.K_s
# KEY_END = pygame.K_e
# KEY_QUIT_RECORDING = pygame.K_q


# class KBReset:
#     def __init__(self):
#         pygame.init()
#         self._screen = pygame.display.set_mode((800, 800))
#         self._set_color(NORMAL)
#         self._saved = False
    
#     def close(self):
#         pygame.display.quit()
#         # TODO: This does not work for Arch Linux
#         pygame.quit()

#     def update(self) -> str:
#         pressed_last = self._get_pressed()

#         if KEY_END in pressed_last:
#             self._set_color(RED)
#             self._saved = False
#             return "end"
        
#         if KEY_QUIT_RECORDING in pressed_last:
#             self._set_color(RED)
#             self._saved = False
#             return "normal"

#         if self._saved:
#             return "save"

#         if KEY_START in pressed_last:
#             self._set_color(GREEN)
#             self._saved = True
#             return "start"

#         self._set_color(NORMAL)
#         return "normal"

#     def _get_pressed(self):
#         pressed = []
#         pygame.event.pump()
#         for event in pygame.event.get():
#             if event.type == pygame.KEYDOWN:
#                 pressed.append(event.key)
#         return pressed

#     def _set_color(self, color):
#         self._screen.fill(color)
#         pygame.display.flip()



@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True

def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["start_recording"] = False
    events["discard_recording"] = False
    events["stop_recording"] = False
    events["quit"] = False

    if is_headless():
        print(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    print("Init!!!!!!")

    def on_press(key):
        try:
            if key == keyboard.KeyCode.from_char("s"):
                events["start_recording"] = True
            elif key == keyboard.KeyCode.from_char("d"):
                events["discard_recording"] = True
            elif key == keyboard.KeyCode.from_char("e"):
                events["stop_recording"] = True
            elif key == keyboard.KeyCode.from_char("q"):
                events["quit"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


# def main():
#     kb = KBReset()
#     while True:
#         state = kb.update()
#         if state == "start":
#             print("start")


# if __name__ == "__main__":
#     main()
