from turtle import distance

import cv2
import time
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from milestone1 import HandDetector
from milestone2 import GestureRecognizer
from milestone3 import VolumeController
from milestone4 import PerformanceMetrics


class GestureApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition Interface")
        self.root.state("zoomed")
        self.root.configure(bg="#0f0f0f")

        # ===== FIXED RIGHT PANEL WIDTH =====
        self.RIGHT_PANEL_WIDTH = 420

        # Grid configuration
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, minsize=self.RIGHT_PANEL_WIDTH)
        self.root.grid_rowconfigure(0, weight=1)

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.camera_running = False
        self.pTime = 0

        self.detection_conf = 0.7
        self.tracking_conf = 0.7
        self.max_hands = 1

        self.detector = HandDetector(
            self.detection_conf,
            self.tracking_conf,
            self.max_hands
        )

        self.gesture = GestureRecognizer()
        self.volume_controller = VolumeController()

        self.metrics = PerformanceMetrics()

        self.build_ui()
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    # ========================= UI =========================

    def build_ui(self):

        # ===== LEFT CAMERA =====
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # ===== RIGHT PANEL =====
        self.panel_container = tk.Frame(
            self.root,
            bg="#0f0f0f",
            width=self.RIGHT_PANEL_WIDTH
        )
        self.panel_container.grid(row=0, column=1, sticky="ns")
        self.panel_container.grid_propagate(False)

        self.canvas = tk.Canvas(
            self.panel_container,
            bg="#0f0f0f",
            highlightthickness=0,
            width=self.RIGHT_PANEL_WIDTH
        )

        self.scrollbar = tk.Scrollbar(
            self.panel_container,
            orient="vertical",
            command=self.canvas.yview
        )

        self.panel = tk.Frame(self.canvas, bg="#0f0f0f")

        self.panel.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window(
            (0, 0),
            window=self.panel,
            anchor="nw",
            width=self.RIGHT_PANEL_WIDTH
        )

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # ===== CARDS =====

        # Controls Card
        control_card = self.create_card("Camera Controls")

        tk.Button(control_card, text="Start Camera",
                  bg="#00bfff", relief="flat",
                  command=self.start_camera).pack(fill="x", pady=5)

        tk.Button(control_card, text="Stop Camera",
                  bg="#ff4444", relief="flat",
                  command=self.stop_camera).pack(fill="x", pady=5)

        tk.Button(control_card, text="Capture",
                  bg="#00ffcc", relief="flat",
                  command=self.capture_image).pack(fill="x", pady=5)

        # Detection Card
        status_card = self.create_card("Detection Status")
        self.status_label = self.create_label("Inactive", "#ff4444", parent=status_card)
        self.hands_label = self.create_label("Hands Detected: 0", parent=status_card)
        self.fps_label = self.create_label("FPS: 0", parent=status_card)
        self.model_label = self.create_label("Model: MediaPipe Hands", parent=status_card)
        self.landmarks_label = self.create_label("Landmarks: 21", parent=status_card)
        self.connections_label = self.create_label("Connections: 21", parent=status_card)

        # Parameters Card
        param_card = self.create_card("Detection Parameters")
        self.create_slider("Detection Confidence", 0, 1,
                           self.detection_conf,
                           self.update_detection_conf,
                           parent=param_card)

        self.create_slider("Tracking Confidence", 0, 1,
                           self.tracking_conf,
                           self.update_tracking_conf,
                           parent=param_card)

        self.create_slider("Max Hands", 1, 4,
                           self.max_hands,
                           self.update_max_hands,
                           is_int=True,
                           parent=param_card)

        # Gesture Card
        gesture_card = self.create_card("Gesture Info")
        self.distance_label = self.create_label("Distance: 0 px", "#00ffcc", 14, parent=gesture_card)
        self.current_gesture_label = self.create_label("Gesture: None", parent=gesture_card)

        # Volume Card
        # ===== Volume Card =====
        volume_card = self.create_card("Volume Control")

        # Percentage Label
        self.volume_label = tk.Label(
            volume_card,
            text="0%",
            bg="#1c1c1c",
            fg="#00ff00",
            font=("Segoe UI", 28, "bold")
        )
        self.volume_label.pack(pady=(10, 5))

        # Volume Bar Canvas
        self.volume_bar_canvas = tk.Canvas(
            volume_card,
            width=60,
            height=200,
            bg="#1c1c1c",
            highlightthickness=0
        )
        self.volume_bar_canvas.pack(pady=10)

        # Background Bar
        self.volume_bar_canvas.create_rectangle(
            20, 10, 40, 190,
            outline="#333333",
            width=2
        )

        # Foreground Bar (dynamic)
        self.volume_bar_fill = self.volume_bar_canvas.create_rectangle(
            20, 190, 40, 190,
            fill="#00ff00",
            width=0
        )

        self.active_label = self.create_label("Active: Yes", "#00ff00", parent=volume_card)
        self.synced_label = self.create_label("Synced: Yes", "#00ff00", parent=volume_card)

        # ================= MILESTONE 4 =================

        metrics_card = self.create_card("Performance Metrics")

        self.metric_volume_label = self.create_label(
            "Volume: 0%",
            parent=metrics_card
        )

        self.metric_distance_label = self.create_label(
            "Finger Distance: 0 px",
            parent=metrics_card
        )

        self.metric_accuracy_label = self.create_label(
            "Accuracy: 100%",
            parent=metrics_card
        )

        self.metric_response_label = self.create_label(
            "Response Time: 0 ms",
            parent=metrics_card
        )

        self.metric_quality_label = self.create_label(
            "Gesture Quality: Good",
            parent=metrics_card
        )

        # Analytics Card
        analytics_card = self.create_card("Analytics")

        # ===== Mapping Graph =====
        self.fig_mapping = Figure(figsize=(3.5, 2.5), dpi=100)
        self.ax_mapping = self.fig_mapping.add_subplot(111)

        self.mapping_canvas = FigureCanvasTkAgg(self.fig_mapping, analytics_card)
        self.mapping_canvas.get_tk_widget().pack(fill="both", padx=10, pady=5)

        # Initial Mapping Graph (before camera starts)
        self.ax_mapping.set_title("Distance → Volume")
        self.ax_mapping.set_xlabel("Distance (%)")
        self.ax_mapping.set_ylabel("Volume (%)")

        self.ax_mapping.set_xlim(0, 100)
        self.ax_mapping.set_ylim(0, 100)

        self.ax_mapping.set_xticks(range(0, 101, 10))
        self.ax_mapping.set_yticks(range(0, 101, 10))

        self.ax_mapping.grid(True)

        self.fig_mapping.tight_layout()
        self.mapping_canvas.draw()

        # ===== History Graph =====
        self.fig_history = Figure(figsize=(3.5, 2.5), dpi=100)
        self.ax_history = self.fig_history.add_subplot(111)

        self.history_canvas = FigureCanvasTkAgg(self.fig_history, analytics_card)
        self.history_canvas.get_tk_widget().pack(fill="both", padx=10, pady=5)

        # Initial History Graph (before camera starts)
        self.ax_history.set_title("Volume History")
        self.ax_history.set_ylim(0, 100)

        self.ax_history.set_xticks(range(0, 21, 5))
        self.ax_history.set_yticks(range(0, 101, 10))

        self.ax_history.grid(True)

        self.fig_history.tight_layout()
        self.history_canvas.draw()

    # ========================= UI Helpers =========================

    def create_card(self, title):
        card = tk.Frame(
            self.panel,
            bg="#1c1c1c",
            highlightbackground="#2a2a2a",
            highlightthickness=1
        )
        card.pack(fill="x", padx=15, pady=10)

        tk.Label(card,
                 text=title,
                 bg="#1c1c1c",
                 fg="#00bfff",
                 font=("Segoe UI", 12, "bold")
                 ).pack(anchor="w", padx=10, pady=(10, 5))
        return card

    def create_label(self, text, color="white", size=11, parent=None):
        label = tk.Label(parent,
                         text=text,
                         bg="#1c1c1c",
                         fg=color,
                         font=("Segoe UI", size))
        label.pack(anchor="w", padx=10, pady=2)
        return label

    def create_slider(self, text, min_val, max_val,
                      default, command, is_int=False, parent=None):

        tk.Label(parent, text=text,
                 bg="#1c1c1c",
                 fg="white").pack(anchor="w", padx=10)

        slider = tk.Scale(parent,
                          from_=min_val,
                          to=max_val,
                          resolution=0.1 if not is_int else 1,
                          orient="horizontal",
                          bg="#1c1c1c",
                          fg="white",
                          troughcolor="#333333",
                          highlightthickness=0,
                          command=command)

        slider.set(default)
        slider.pack(fill="x", padx=10, pady=5)

    # ========================= Slider Updates =========================

    def update_detection_conf(self, val):
        self.detection_conf = float(val)
        self.detector.init_model(self.detection_conf,
                                 self.tracking_conf,
                                 self.max_hands)

    def update_tracking_conf(self, val):
        self.tracking_conf = float(val)
        self.detector.init_model(self.detection_conf,
                                 self.tracking_conf,
                                 self.max_hands)

    def update_max_hands(self, val):
        self.max_hands = int(float(val))
        self.detector.init_model(self.detection_conf,
                                 self.tracking_conf,
                                 self.max_hands)

    # ========================= Camera =========================

    def start_camera(self):
        self.camera_running = True
        self.update_frame()

    def stop_camera(self):
        self.camera_running = False
        self.status_label.config(text="Inactive", fg="#ff4444")

    def capture_image(self):
        if hasattr(self, "current_frame"):
            cv2.imwrite("captured_frame.jpg", self.current_frame)

    # ========================= Frame Loop =========================

    def update_frame(self):

        if not self.camera_running:
            return

        success, frame = self.cap.read()
        if not success:
            return

        frame = cv2.flip(frame, 1)
        results = self.detector.detect(frame)

        hands_detected = 0
        distance = 0
        gesture_name = None

        if results.multi_hand_landmarks:
            hands_detected = len(results.multi_hand_landmarks)
            self.status_label.config(text="Active", fg="#00ff00")

            for hand in results.multi_hand_landmarks:
                self.detector.draw(frame, hand)

                distance, gesture_name, coords = \
                    self.gesture.calculate_distance(hand, frame.shape)

                x1, y1, x2, y2 = coords
                cv2.circle(frame, (x1, y1), 8, (255, 0, 0), -1)
                cv2.circle(frame, (x2, y2), 8, (255, 0, 0), -1)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                normalized = self.volume_controller.map_distance_to_volume(distance)
                volume_percent = self.volume_controller.set_volume(normalized)

                self.volume_label.config(text=f"{volume_percent}%")

                self.last_distance = int(distance)
                self.last_volume = volume_percent

                self.update_graphs()

                # ===== Milestone 4 Metrics =====

                response = self.metrics.update()
                quality = self.metrics.evaluate_gesture_quality(distance)

                self.metric_volume_label.config(text=f"Volume: {volume_percent}%")
                self.metric_distance_label.config(text=f"Finger Distance: {int(distance)} px")
                self.metric_accuracy_label.config(text="Accuracy: 100%")
                self.metric_response_label.config(text=f"Response Time: {response} ms")
                self.metric_quality_label.config(text=f"Gesture Quality: {quality}")

                # Update Volume Bar
                bar_height = 180  # total drawable height
                fill_height = (volume_percent / 100) * bar_height

                self.volume_bar_canvas.coords(
                self.volume_bar_fill,
                    20,
                    190 - fill_height,
                    40,
                    190
                )

        self.distance_label.config(text=f"Distance: {int(distance)} px")

        if gesture_name:
            self.current_gesture_label.config(
                text=f"Gesture: {gesture_name}",
                fg="#00ff00"
            )
        else:
            self.current_gesture_label.config(
                text="Gesture: None",
                fg="white"
            )

        cTime = time.time()
        fps = 1 / (cTime - self.pTime) if (cTime - self.pTime) != 0 else 0
        self.pTime = cTime

        self.hands_label.config(text=f"Hands Detected: {hands_detected}")
        self.fps_label.config(text=f"FPS: {int(fps)}")

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.current_frame = frame
        self.root.after(10, self.update_frame)

    def update_graphs(self):

        distance = getattr(self, "last_distance", 0)
        volume = getattr(self, "last_volume", 0)

        # ===== Mapping Graph =====
        self.ax_mapping.clear()

        min_d = self.volume_controller.min_dist
        max_d = self.volume_controller.max_dist

        # Normalize distance → 0–100
        normalized_distance = int(
            ((distance - min_d) / (max_d - min_d)) * 100
        )
        normalized_distance = max(0, min(100, normalized_distance))

        # Ideal linear mapping (0–100)
        x = list(range(0, 101))
        y = list(range(0, 101))

        self.ax_mapping.plot(x, y, label="Mapping", linewidth=2)

        # Current position (normalized)
        self.ax_mapping.scatter(
            normalized_distance,
            volume,
            color="red",
            s=80,
            label="Current"
        )

        # Axis settings (clean 0–100 scale)
        self.ax_mapping.set_xlim(0, 100)
        self.ax_mapping.set_ylim(0, 100)

        # Proper ticks (0,10,20,...100)
        self.ax_mapping.set_xticks(range(0, 101, 10))
        self.ax_mapping.set_yticks(range(0, 101, 10))

        self.ax_mapping.set_title("Distance → Volume")
        self.ax_mapping.set_xlabel("Distance (%)")
        self.ax_mapping.set_ylabel("Volume (%)")

        self.ax_mapping.grid(True)
        self.ax_mapping.legend()

        # Fix cropped labels
        self.fig_mapping.tight_layout()

        self.mapping_canvas.draw()

        # ===== History Graph =====
        self.ax_history.clear()

        history = self.volume_controller.history or [0]

        self.ax_history.bar(range(len(history)), history)
        self.ax_history.set_ylim(0, 100)
        self.ax_history.set_title("Volume History")

        self.fig_mapping.tight_layout()

        self.history_canvas.draw()

    # ========================= Scroll =========================

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


root = tk.Tk()
app = GestureApp(root)
root.mainloop()