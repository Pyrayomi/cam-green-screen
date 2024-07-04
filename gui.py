import tkinter as tk
from tkinter import ttk, filedialog
from virtual_green_screen import VirtualGreenScreen
from PIL import Image, ImageTk
import cv2

VALID_RESOLUTIONS = [
    "426x240", "640x360", "854x480", "1280x720", "1366x768", "1600x900", "1920x1080", "2560x1440", "3840x2160"
]


class GUI(tk.Tk):
    def __init__(self, green_screen):
        """
        Inicializa a classe GUI com a instância do VirtualGreenScreen.

        Args:
            green_screen (VirtualGreenScreen): Instância da classe VirtualGreenScreen.
        """
        super().__init__()
        self.green_screen = green_screen
        self.title("Virtual Green Screen Settings")
        self.geometry("800x600")

        # Track previous settings
        self.previous_settings = {
            'resolution': (self.green_screen.cam_width, self.green_screen.cam_height),
            'gif_path': self.green_screen.gif_path,
            'frame_delay': self.green_screen.frame_delay,
            'keep_aspect_ratio': self.green_screen.keep_aspect_ratio
        }

        self.create_widgets()
        self.update_video()

    def create_widgets(self):
        """Cria os widgets da interface gráfica."""
        self.label_aspect_ratio = tk.Label(self, text="Keep Aspect Ratio")
        self.label_aspect_ratio.pack()

        self.keep_aspect_ratio_var = tk.BooleanVar(value=self.green_screen.keep_aspect_ratio)
        self.checkbox_aspect_ratio = tk.Checkbutton(self, variable=self.keep_aspect_ratio_var,
                                                    command=self.update_settings)
        self.checkbox_aspect_ratio.pack()

        self.label_resolution = tk.Label(self, text="Resolution")
        self.label_resolution.pack()

        self.resolution_var = tk.StringVar(value=f"{self.green_screen.target_width}x{self.green_screen.target_height}")
        self.combobox_resolution = ttk.Combobox(self, textvariable=self.resolution_var, values=VALID_RESOLUTIONS)
        self.combobox_resolution.pack()

        self.label_frame_delay = tk.Label(self, text="Frame Delay")
        self.label_frame_delay.pack()

        self.frame_delay_var = tk.IntVar(value=self.green_screen.frame_delay)
        self.spinbox_frame_delay = tk.Spinbox(self, from_=1, to=10, textvariable=self.frame_delay_var,
                                              command=self.update_settings)
        self.spinbox_frame_delay.pack()

        self.label_gif = tk.Label(self, text="GIF Path")
        self.label_gif.pack()

        self.gif_path_var = tk.StringVar(value=self.green_screen.gif_path)
        self.entry_gif_path = tk.Entry(self, textvariable=self.gif_path_var)
        self.entry_gif_path.pack()

        self.browse_gif_button = tk.Button(self, text="Browse", command=self.browse_gif)
        self.browse_gif_button.pack()

        self.apply_button = tk.Button(self, text="Apply", command=self.apply_changes)
        self.apply_button.pack()

        self.start_button = tk.Button(self, text="Start", command=self.start_green_screen)
        self.start_button.pack()

        self.stop_button = tk.Button(self, text="Stop", command=self.stop_green_screen)
        self.stop_button.pack()

        self.video_label = tk.Label(self)
        self.video_label.pack()

    def browse_gif(self):
        """Abre um diálogo para selecionar um arquivo GIF e atualiza o caminho do GIF."""
        gif_path = filedialog.askopenfilename(filetypes=[("GIF, WEBP files", "*.gif *.webp"), ("GIF files", "*.gif"), ("WEBP files", "*.webp"), ("All files", "*.*")]
)
        if gif_path:
            self.gif_path_var.set(gif_path)
            self.update_settings()

    def update_settings(self):
        """Atualiza as configurações do green screen com base nos valores da interface gráfica."""
        self.green_screen.keep_aspect_ratio = self.keep_aspect_ratio_var.get()
        resolution = self.resolution_var.get()
        if resolution in VALID_RESOLUTIONS:
            width, height = map(int, resolution.split('x'))
            self.green_screen.cam_width = width
            self.green_screen.cam_height = height
        self.green_screen.frame_delay = self.frame_delay_var.get()
        self.green_screen.gif_path = self.gif_path_var.get()

    def apply_changes(self):
        """Aplica as mudanças nas configurações e atualiza os valores anteriores."""
        self.update_settings()
        self.green_screen.apply_changes(self.previous_settings)
        self.previous_settings = {
            'resolution': (self.green_screen.cam_width, self.green_screen.cam_height),
            'gif_path': self.green_screen.gif_path,
            'frame_delay': self.green_screen.frame_delay,
            'keep_aspect_ratio': self.green_screen.keep_aspect_ratio
        }

    def start_green_screen(self):
        """Inicia a execução do green screen virtual."""
        self.green_screen.run()

    def stop_green_screen(self):
        """Para a execução do green screen virtual."""
        self.green_screen.stop()

    def update_video(self):
        """Atualiza o vídeo exibido na interface gráfica."""
        if self.green_screen.frame is not None:
            frame = self.green_screen.frame  # Certifica que o quadro está no formato uint8
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Corrige BGR para RGB para tkinter
            frame = ImageTk.PhotoImage(frame)
            self.video_label.config(image=frame)
            self.video_label.image = frame
        self.after(10, self.update_video)  # Atualiza a cada 10 ms


if __name__ == "__main__":
    gif_path = r'C:\Users\rayom\OneDrive\Imagens\dog3.webp'
    green_screen = VirtualGreenScreen(gif_path, cam_width=426, cam_height=240, frame_delay=2, keep_aspect_ratio=False)
    gui = GUI(green_screen)
    gui.mainloop()
