import cv2
import mediapipe as mp
import pyvirtualcam
from PIL import Image, ImageSequence
import numpy as np
import threading


class VirtualGreenScreen:
    def __init__(self, gif_path, cam_width=1280, cam_height=720, fps=30, frame_delay=3, keep_aspect_ratio=True):
        """
        Inicializa a classe VirtualGreenScreen com os parâmetros fornecidos.

        Args:
            gif_path (str): Caminho para o arquivo GIF.
            cam_width (int): Largura da câmera.
            cam_height (int): Altura da câmera.
            fps (int): Frames por segundo.
            frame_delay (int): Intervalo de atraso dos quadros do GIF.
            keep_aspect_ratio (bool): Manter a proporção do GIF.
        """
        self.gif_path = gif_path
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.fps = fps
        self.frame_delay = frame_delay
        self.keep_aspect_ratio = keep_aspect_ratio
        self.running = False
        self.frame = None
        self.initialize_resources()

    def initialize_resources(self):
        """Inicializa os recursos necessários, incluindo a câmera e o MediaPipe."""
        self.target_width = self.cam_width
        self.target_height = self.cam_height
        self.cap = self.initialize_webcam(self.target_width, self.target_height)
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.gif_frames = self.load_gif(self.gif_path)
        self.gif_frame_count = len(self.gif_frames)
        self.gif_index = 0
        self.frame_counter = 0
        self.prev_mask = None

    def load_gif(self, gif_path):
        """
        Carrega e converte o GIF em uma lista de quadros.

        Args:
            gif_path (str): Caminho para o arquivo GIF.

        Returns:
            list: Lista de quadros do GIF como arrays numpy.
        """
        gif = Image.open(gif_path)
        frames = [np.array(frame.convert('RGB')) for frame in ImageSequence.Iterator(gif)]
        return frames

    def initialize_webcam(self, width, height):
        """
        Inicializa a captura de vídeo da webcam com a resolução especificada.

        Args:
            width (int): Largura da captura de vídeo.
            height (int): Altura da captura de vídeo.

        Returns:
            cv2.VideoCapture: Objeto de captura de vídeo.
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    def process_frame_with_mediapipe(self, frame, alpha=0.5):
        """
        Processa o quadro com MediaPipe para obter a máscara de segmentação.

        Args:
            frame (np.array): Quadro de entrada.
            alpha (float): Fator de suavização para estabilização temporal.

        Returns:
            tuple: Quadro RGB processado e máscara de segmentação suavizada.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_selfie_segmentation.process(rgb_frame)
        mask = results.segmentation_mask > 0.5

        # Suavização da máscara
        mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), sigmaX=0, sigmaY=0)
        mask = cv2.medianBlur((mask * 255).astype(np.uint8), 5) / 255.0

        # Estabilização temporal
        if self.prev_mask is not None:
            mask = cv2.addWeighted(mask, alpha, self.prev_mask, 1 - alpha, 0)

        return rgb_frame, mask

    def resize_and_center_gif(self, gif_frame):
        """
        Redimensiona o GIF mantendo a proporção (se configurado) e centraliza-o.

        Args:
            gif_frame (np.array): Quadro do GIF.

        Returns:
            np.array: Quadro do GIF redimensionado e centralizado.
        """
        if self.keep_aspect_ratio:
            return self._resize_with_aspect_ratio(gif_frame)
        else:
            return cv2.resize(gif_frame, (self.target_width, self.target_height))

    def _resize_with_aspect_ratio(self, gif_frame):
        """
        Redimensiona o GIF mantendo a proporção e centraliza-o.

        Args:
            gif_frame (np.array): Quadro do GIF.

        Returns:
            np.array: Quadro do GIF redimensionado e centralizado.
        """
        gif_height, gif_width, _ = gif_frame.shape
        aspect_ratio = gif_width / gif_height

        if self.target_width / self.target_height > aspect_ratio:
            new_height = self.target_height
            new_width = int(aspect_ratio * new_height)
        else:
            new_width = self.target_width
            new_height = int(new_width / aspect_ratio)

        resized_gif = cv2.resize(gif_frame, (new_width, new_height))

        centered_gif = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        x_offset = (self.target_width - new_width) // 2
        y_offset = (self.target_height - new_height) // 2

        centered_gif[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_gif

        return centered_gif

    def combine_frames(self, rgb_frame, gif_frame, mask):
        """
        Combina o quadro da câmera com o quadro do GIF usando a máscara.

        Args:
            rgb_frame (np.array): Quadro RGB da câmera.
            gif_frame (np.array): Quadro do GIF.
            mask (np.array): Máscara de segmentação.

        Returns:
            np.array: Quadro combinado.
        """
        mask_3d = np.repeat(mask[..., np.newaxis], 3, axis=2)
        gif_frame = cv2.resize(gif_frame, (
        rgb_frame.shape[1], rgb_frame.shape[0]))  # Certifica que gif_frame tem as mesmas dimensões que rgb_frame
        return rgb_frame * mask_3d + gif_frame * (1 - mask_3d)

    def run(self):
        """Inicia a execução do green screen virtual em uma nova thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _run(self):
        """Método interno que executa a lógica do green screen virtual."""
        self.initialize_resources()

        ret, orig = self.cap.read()
        if not ret:
            print("Erro ao acessar a webcam.")
            self.running = False
            return

        self.cam = pyvirtualcam.Camera(width=self.target_width, height=self.target_height, fps=self.fps)
        print(f'Using virtual camera: {self.cam.device}')

        while self.running:
            if not self.cap.isOpened():
                self.cap = self.initialize_webcam(self.target_width, self.target_height)

            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame, mask = self.process_frame_with_mediapipe(frame)
            self.prev_mask = mask

            if self.frame_counter % self.frame_delay == 0:
                self.gif_index = (self.gif_index + 1) % self.gif_frame_count
            self.frame_counter += 1

            gif_frame = self.resize_and_center_gif(self.gif_frames[self.gif_index])
            combined_frame = self.combine_frames(rgb_frame, gif_frame, mask)

            # Redimensiona o quadro combinado para corresponder à resolução alvo
            combined_frame_resized = cv2.resize(combined_frame, (self.target_width, self.target_height))

            self.cam.send(combined_frame_resized.astype(np.uint8))
            self.cam.sleep_until_next_frame()

            self.frame = combined_frame_resized.astype(np.uint8)  # Garante que o quadro está no formato uint8

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.cap.release()
        self.cam.close()
        cv2.destroyAllWindows()

    def stop(self):
        """Para a execução do green screen virtual e libera recursos."""
        self.running = False
        if hasattr(self, 'thread') and self.thread is not None:
            self.thread.join()
        self.cap.release()
        if hasattr(self, 'cam') and self.cam is not None:
            self.cam.close()
        cv2.destroyAllWindows()

    def apply_changes(self, previous_settings):
        """
        Aplica mudanças baseadas nas configurações atualizadas em comparação com as configurações anteriores.

        Args:
            previous_settings (dict): Dicionário contendo as configurações anteriores.
        """
        if previous_settings['resolution'] != (self.cam_width, self.cam_height):
            self.stop()
            self.initialize_resources()
            self.run()
        if previous_settings['gif_path'] != self.gif_path:
            self.gif_frames = self.load_gif(self.gif_path)
            self.gif_frame_count = len(self.gif_frames)
            self.gif_index = 0
        if previous_settings['frame_delay'] != self.frame_delay:
            pass  # Nenhuma inicialização específica necessária para frame delay
        if previous_settings['keep_aspect_ratio'] != self.keep_aspect_ratio:
            pass  # Nenhuma inicialização específica necessária para manter a proporção
