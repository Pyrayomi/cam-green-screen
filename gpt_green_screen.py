import cv2
import mediapipe as mp
import pyvirtualcam
from PIL import Image, ImageSequence
import numpy as np


class VirtualGreenScreen:
    def __init__(self, gif_path, cam_width=1280, cam_height=720, fps=30, frame_delay=3, keep_aspect_ratio=True):
        self.cap = self.initialize_webcam(cam_width, cam_height)
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.gif_frames = self.load_gif(gif_path)
        self.gif_frame_count = len(self.gif_frames)
        self.gif_index = 0
        self.frame_delay = frame_delay
        self.frame_counter = 0
        self.prev_mask = None
        self.target_width = cam_width
        self.target_height = cam_height
        self.fps = fps
        self.keep_aspect_ratio = keep_aspect_ratio

    def load_gif(self, gif_path):
        """Carregar e converter o GIF em uma lista de quadros."""
        gif = Image.open(gif_path)
        frames = [np.array(frame.convert('RGB')) for frame in ImageSequence.Iterator(gif)]
        return frames

    def initialize_webcam(self, width, height):
        """Inicializar a captura de vídeo da webcam com a resolução especificada."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    def process_frame_with_mediapipe(self, frame, alpha=0.5):
        """Processar o quadro com MediaPipe para obter a máscara de segmentação."""
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
        """Redimensionar o GIF mantendo a proporção e centralizá-lo."""
        if self.keep_aspect_ratio:
            gif_height, gif_width, _ = gif_frame.shape
            aspect_ratio = gif_width / gif_height

            if self.target_width / self.target_height > aspect_ratio:
                new_height = self.target_height
                new_width = int(aspect_ratio * new_height)
            else:
                new_width = self.target_width
                new_height = int(new_width / aspect_ratio)

            resized_gif = cv2.resize(gif_frame, (new_width, new_height))

            # Criar uma imagem preta com o tamanho da tela
            centered_gif = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
            x_offset = (self.target_width - new_width) // 2
            y_offset = (self.target_height - new_height) // 2

            # Colocar o GIF redimensionado na imagem preta centralizada
            centered_gif[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_gif

            return centered_gif
        else:
            return cv2.resize(gif_frame, (self.target_width, self.target_height))

    def combine_frames(self, rgb_frame, gif_frame, mask):
        """Combinar o quadro da câmera com o quadro do GIF usando a máscara."""
        mask_3d = np.repeat(mask[..., np.newaxis], 3, axis=2)
        return rgb_frame * mask_3d + gif_frame * (1 - mask_3d)

    def run(self):
        ret, orig = self.cap.read()
        if not ret:
            print("Erro ao acessar a webcam.")
            return

        with pyvirtualcam.Camera(width=self.target_width, height=self.target_height, fps=self.fps) as cam:
            print(f'Using virtual camera: {cam.device}')

            while True:
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

                cam.send(combined_frame.astype(np.uint8))
                cam.sleep_until_next_frame()

                bgr_combined_frame = cv2.cvtColor(combined_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imshow('Green Screen Output', bgr_combined_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    gif_path = r'dog3.webp'
    green_screen = VirtualGreenScreen(gif_path, cam_width=640, cam_height=480, frame_delay=2, keep_aspect_ratio=False)
    green_screen.run()
