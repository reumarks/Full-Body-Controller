import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pynput import keyboard
from pynput.keyboard import Controller, Key

# Constants
DEFAULT_ZONE_SIZE = 100
DEFAULT_THRESHOLD = 0.5
HANDLE_RADIUS = 8
SLIDER_HEIGHT = 8
OVERLAY_ALPHA = 0.3
DEFAULT_COLOR_THRESHOLD = 50

REGISTERED_MOVES = {
    'Up': ['up'],
    'Down': ['down'],
    'Jump': ['z'],
    'Attack': ['x'],
    'Sprint': ['c'],
    'Bind': ['a'],
    'Inventory': ['i'],
    'Left': ['left'],
    'Right': ['right'],
    'Map':['tab'],
    'Harpoon': ['s'],
    'Needolin': ['d'],
    'Tool': ['f'],
    
    'Jump+R': ['right', 'z'],
    'Jump+L': ['left', 'z'],
}

@dataclass
class Zone:
    x: int
    y: int
    size: int = DEFAULT_ZONE_SIZE
    threshold: float = DEFAULT_THRESHOLD
    move: Optional[str] = None
    active: bool = False
    green_count: int = 0
    max_pixels: Optional[int] = None
    bounds: Optional[Tuple[int, int, int, int]] = None
    
    def update_bounds(self):
        half = self.size >> 1
        self.bounds = (self.x - half, self.y - half, self.x + half, self.y + half)
        self.max_pixels = self.size * self.size
    
    def contains_point(self, px: int, py: int) -> bool:
        half = self.size >> 1
        return abs(px - self.x) <= half and abs(py - self.y) <= half


class ColorPicker:
    def __init__(self):
        self.reference_color = np.array([0, 255, 0], dtype=np.uint8)
        self.color_threshold = DEFAULT_COLOR_THRESHOLD
        self.picking_mode = False
        self.hover_color = None
        self.slider_dragging = False
    
    def handle_click(self, frame, x: int, y: int) -> bool:
        # Clicking mode
        if self.picking_mode:
            self.pick_color(frame, x, y)
            return True
                
        # Color picker clicked
        if abs(x - 1875) <= 20 and abs(y - 35) <= 20:
            self.picking_mode = True
            return True
        
        # Clicking slider
        if abs(x - 1875) <= 10 and 80 <= y <= 220:
            self.slider_dragging = True
            self.update_threshold_from_position(y)
            return True
        
        return False
    
    def handle_move(self, x: int, y: int, frame: np.ndarray):
        if self.slider_dragging:
            self.update_threshold_from_position(y)
        elif self.picking_mode and frame is not None:
            h, w = frame.shape[:2]
            if 0 <= y < h and 0 <= x < w:
                self.hover_color = frame[y, x].copy()
    
    def update_threshold_from_position(self, y: int):
        ratio = constrain((y - 80) / (220 - 80), 0.0, 1.0)
        self.color_threshold = int(ratio * 255)
    
    def release_slider(self):
        self.slider_dragging = False
    
    def pick_color(self, frame: np.ndarray, x: int, y: int):
        if self.picking_mode:
            h, w = frame.shape[:2]
            if 0 <= y < h and 0 <= x < w:
                self.reference_color = frame[y, x].copy()
            self.picking_mode = False
            self.hover_color = None
    
    def check_color_similarity(self, roi: np.ndarray) -> np.ndarray:
        diff = np.abs(roi.astype(np.float32) - self.reference_color.astype(np.float32))
        distance = np.sqrt(np.sum(diff ** 2, axis=2))
        return distance <= self.color_threshold
    
    def draw(self, frame: np.ndarray, mouse_x: int = 0, mouse_y: int = 0):
        color_bgr = tuple(int(c) for c in self.reference_color)
        cv2.rectangle(frame, (1855, 15), (1895, 55), color_bgr, -1)
        cv2.rectangle(frame, (1855, 15), (1895, 55), (255, 255, 255), 2)
        cv2.line(frame, (1875, 80), (1875, 220), (180, 180, 180), 4)
        
        ratio = self.color_threshold / 255.0
        handle_y = int(80 + ratio * 140)
        
        cv2.circle(frame, (1875, handle_y), 12, (220, 220, 220), -1)
        
        if self.picking_mode and self.hover_color is not None:
            hover_bgr = tuple(int(c) for c in self.hover_color)
            cv2.circle(frame, (mouse_x + 20, mouse_y - 20), 15, hover_bgr, -1)
            cv2.circle(frame, (mouse_x + 20, mouse_y - 20), 15, (255, 255, 255), 1)


class ModeManager:
    def __init__(self):
        self.setup_mode = True
        self.listener = keyboard.GlobalHotKeys({'`': self.toggle_mode})
        self.listener.start()
    
    def toggle_mode(self):
        self.setup_mode = not self.setup_mode
    
    def cleanup(self):
        self.listener.stop()


class CameraManager:
    def __init__(self):
        self.cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    
    def read_frame(self):
        self.cap.grab()
        ret, frame = self.cap.retrieve()
        return cv2.flip(frame, 1) if ret else None
    
    def release(self):
        self.cap.release()

class SetupModeHandler:
    def __init__(self, zones: List[Zone], color_picker: ColorPicker):
        self.zones = zones
        self.color_picker = color_picker
        self.selected: Optional[Zone] = None
        self.dragging = False
        self.drag_offset = (0, 0)
        self.resizing = False
        self.resize_start = None
        self.slider_dragging = False
        self.mouse_x = 0
        self.mouse_y = 0
        self.create_zones()
    
    def create_zones(self):
        for index, (current_move, _) in enumerate(REGISTERED_MOVES.items()):
            zone = Zone(150 + 150 * (index % 5), 150 + 150 * math.floor(index / 5))
            zone.update_bounds()
            zone.move = current_move
            self.zones.append(zone)
    
    def handle_mouse(self, event: int, x: int, y: int, frame: np.ndarray = None):
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down(x, y, frame)
        elif event == cv2.EVENT_MOUSEMOVE:
            self._mouse_move(x, y, frame)
        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_up(x, y, frame)
    
    def _mouse_down(self, x: int, y: int, frame: np.ndarray = None):
        # Check color picker and slider
        if self.color_picker.handle_click(frame, x, y):
            return
        
        if self.selected:
            x1, y1, x2, y2 = self.selected.bounds
            
            # Delete button
            if (x - x2) ** 2 + (y - y1) ** 2 <= 144:
                self.zones.remove(self.selected)
                self.selected = None
                return
            
            # Slider
            sx = self.selected.x - (self.selected.size >> 1) - 14
            if sx - 6 <= x <= sx + 6 and y1 <= y <= y2:
                self.slider_dragging = True
                return
            
            # Resize handle
            if (x - x2) ** 2 + (y - y2) ** 2 <= HANDLE_RADIUS ** 2:
                self.resizing = True
                self.resize_start = (x, y, self.selected.size)
                return
        
        # Drag zone
        for zone in self.zones:
            if zone.contains_point(x, y):
                self.selected = zone
                self.dragging = True
                self.drag_offset = (x - zone.x, y - zone.y)
                return
            
        self.selected = None
    
    def _mouse_move(self, x: int, y: int, frame: np.ndarray = None):
        # Handle color picker
        self.color_picker.handle_move(x, y, frame)
        
        # Move zone
        if self.dragging and self.selected:
            self.selected.x = x - self.drag_offset[0]
            self.selected.y = y - self.drag_offset[1]
            self.selected.update_bounds()
        
        # Resize zone
        elif self.resizing and self.selected and self.resize_start:
            sx, sy, start_size = self.resize_start
            self.selected.size = max(50, start_size + (max(x - sx, y - sy) << 1))
            self.selected.update_bounds()
        
        # Update zone threshold
        elif self.slider_dragging and self.selected:
            self.selected.threshold = constrain((self.selected.bounds[3] - y) / self.selected.size, 0.0, 1.0)
    
    def _mouse_up(self, x: int, y: int, frame: np.ndarray = None):
        self.dragging = False
        self.resizing = False
        self.slider_dragging = False
        self.resize_start = None
        self.color_picker.release_slider()
    
    def render(self, frame: np.ndarray, debug_frame: np.ndarray):
        for zone in self.zones:
            self._draw_zone(frame, debug_frame, zone, zone == self.selected)
        
        self.color_picker.draw(frame, self.mouse_x, self.mouse_y)
    
    def _draw_zone(self, frame: np.ndarray, debug_frame: np.ndarray, zone: Zone, selected: bool):
        x1, y1, x2, y2 = zone.bounds
        
        # In setup mode, overlay green on matching pixels
        if x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
            roi = debug_frame[y1:y2, x1:x2]
            mask = self.color_picker.check_color_similarity(roi)
            
            # Create green overlay
            overlay = frame[y1:y2, x1:x2].copy()
            overlay[mask] = (0, 255, 0)
            
            # Blend overlay
            cv2.addWeighted(overlay, 0.5, frame[y1:y2, x1:x2], 0.5, 0, frame[y1:y2, x1:x2])
        
        # Border
        color = (100, 255, 100) if zone.active else (200, 200, 200)
        thickness = 3 if selected else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Key label
        if zone.move:
            text = zone.move
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(frame, text, (zone.x - (tw >> 1), zone.y + (th >> 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Progress bar
        max_px = zone.max_pixels
        slider_pos = int(y2 - zone.threshold * zone.size)
        bar_height = int((x2 - x1 - 4) * constrain(zone.green_count / max_px, 0.0, 1.0))
        bar_color = (100, 255, 100) if zone.active else (200, 200, 200)

        cv2.line(frame, (x1, slider_pos), (x1 + 8, slider_pos), (100, 255, 100), 2)
        cv2.rectangle(frame, (x1 + 2, y2 - 2 - bar_height), (x1 + 6, y2 - 2), bar_color, -1)
        
        if selected:
            # Resize handle
            cv2.circle(frame, (x2, y2), 12, (220, 220, 220), -1)
            cv2.line(frame, (x2 - 5, y2 - 5), (x2 + 5, y2 + 5), (80, 80, 80), 2)
            cv2.line(frame, (x2 + 5, y2 + 1), (x2 + 5, y2 + 5), (80, 80, 80), 2)
            cv2.line(frame, (x2 + 1, y2 + 5), (x2 + 5, y2 + 5), (80, 80, 80), 2)
            cv2.line(frame, (x2 - 5, y2 - 1), (x2 - 5, y2 - 5), (80, 80, 80), 2)
            cv2.line(frame, (x2 - 1, y2 - 5), (x2 - 5, y2 - 5), (80, 80, 80), 2)
            
            # Threshold slider
            cv2.line(frame, (x1 - 14, y1 + 2), (x1 - 14, y2 - 2), (180, 180, 180), 4)
            cv2.circle(frame, (x1 - 14, slider_pos), 12, (220, 220, 220), -1)

            # Delete button
            cv2.circle(frame, (x2, y1), 12, (220, 220, 220), -1)
            cv2.line(frame, (x2 - 5, y1 - 5), (x2 + 5, y1 + 5), (80, 80, 80), 2)
            cv2.line(frame, (x2 - 5, y1 + 5), (x2 + 5, y1 - 5), (80, 80, 80), 2)

class RecordingModeHandler:
    def __init__(self, zones: List[Zone], color_picker: ColorPicker):
        self.zones = zones
        self.color_picker = color_picker
        self.keys_pressed = {}
        self.keyboard = Controller()

        for move_keys in REGISTERED_MOVES.values():
            for key in move_keys:
                self.keys_pressed[key] = False

    def _key(self, name):
        # Map human names like 'up' or 'left' to pynput Keys
        mapping = {
            'up': Key.up,
            'down': Key.down,
            'left': Key.left,
            'right': Key.right,
            'tab': Key.tab,
        }
        return mapping.get(name, name)  # default: normal character

    def process(self, frame: np.ndarray, recording_mode):
        keys_to_press = set()

        for zone in self.zones:
            x1, y1, x2, y2 = zone.bounds
            roi = frame[y1:y2, x1:x2]
            mask = self.color_picker.check_color_similarity(roi)
            zone.green_count = np.count_nonzero(mask)
            threshold_px = zone.threshold * zone.max_pixels
            zone.active = zone.green_count > threshold_px

            if recording_mode and zone.active and zone.move:
                for key in REGISTERED_MOVES[zone.move]:
                    keys_to_press.add(key)

        if recording_mode:
            # Press keys that should be down but aren’t
            for key in keys_to_press:
                if not self.keys_pressed[key]:
                    self.keyboard.press(self._key(key))
                    self.keys_pressed[key] = True

            # Release keys that are no longer active
            for key in list(self.keys_pressed):
                if self.keys_pressed[key] and key not in keys_to_press:
                    self.keyboard.release(self._key(key))
                    self.keys_pressed[key] = False

    def release_all(self):
        for key, pressed in self.keys_pressed.items():
            if pressed:
                self.keyboard.release(self._key(key))
                self.keys_pressed[key] = False
            
    def render(self, frame: np.ndarray):
        for zone in self.zones:
            self._draw_zone(frame, zone)
    
    def _draw_zone(self, frame: np.ndarray, zone: Zone):
        x1, y1, x2, y2 = zone.bounds
        
        # Border
        color = (100, 255, 100) if zone.active else (200, 200, 200)
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Key label
        if zone.move:
            text = zone.move
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(frame, text, (zone.x - (tw >> 1), zone.y + (th >> 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def transform_coords(x: int, y: int, offset: Tuple[int, int], display_size: Tuple[int, int], frame_size: Tuple[int, int]) -> Tuple[Optional[int], Optional[int]]:
    x_off, y_off = offset
    disp_w, disp_h = display_size
    frame_w, frame_h = frame_size
    
    rel_x, rel_y = x - x_off, y - y_off
    
    if rel_x < 0 or rel_y < 0 or rel_x >= disp_w or rel_y >= disp_h:
        return None, None
    
    return int(rel_x * frame_w / disp_w), int(rel_y * frame_h / disp_h)

def letterbox(frame: np.ndarray, window_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    h, w = frame.shape[:2]
    win_w, win_h = window_size
    
    scale = min(win_w / w, win_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    x_off, y_off = (win_w - new_w) // 2, (win_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    
    return canvas, (x_off, y_off), (new_w, new_h)

def constrain(x: float, x_min: float, x_max: float) -> float:
    return min(max(x, x_min), x_max)

def main():
    zones = []
    color_picker = ColorPicker()
    mode_mgr = ModeManager()
    camera = CameraManager()
    setup = SetupModeHandler(zones, color_picker)
    recording = RecordingModeHandler(zones, color_picker)
    
    cv2.namedWindow('Full Body Controller', cv2.WINDOW_NORMAL)
    
    offset, disp_size, frame_size = (0, 0), (0, 0), (0, 0)
    current_frame = None
    
    def mouse_cb(event, x, y, flags, param):
        nonlocal current_frame
        tx, ty = transform_coords(x, y, offset, disp_size, frame_size)
        if tx is not None and ty is not None:
            setup.handle_mouse(event, tx, ty, current_frame)
    
    cv2.setMouseCallback('Full Body Controller', mouse_cb)
    
    try:
        while True:
            frame = camera.read_frame()

            if frame is None:
                break
            
            h, w = frame.shape[:2]
            frame_size = (w, h)
            
            # Store frame for color picking
            current_frame = frame.copy()
            
            # Process zones
            recording.process(frame, not mode_mgr.setup_mode)
            
            # Render UI
            if mode_mgr.setup_mode:
                setup.render(frame, current_frame)
            else:
                recording.render(frame)
            
            # Mode indicator
            mode_text = "SETUP" if mode_mgr.setup_mode else "RECORDING"
            mode_color = (180, 180, 180) if mode_mgr.setup_mode else (100, 255, 100)
            (tw, th), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (15, h - 35), (25 + tw, h - 15), (40, 40, 40), -1)
            cv2.putText(frame, mode_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
            
            # Display
            rect = cv2.getWindowImageRect('Full Body Controller')
            win_w, win_h = rect[2], rect[3]
            
            if win_w > 0 and win_h > 0:
                display, offset, disp_size = letterbox(frame, (win_w, win_h))
                cv2.imshow('Full Body Controller', display)
            else:
                cv2.imshow('Full Body Controller', frame)
            
            key = cv2.waitKey(1)
            
            if cv2.getWindowProperty('Full Body Controller', cv2.WND_PROP_VISIBLE) < 1 or key == 27:
                break
    
    finally:
        recording.release_all()
        camera.release()
        mode_mgr.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()