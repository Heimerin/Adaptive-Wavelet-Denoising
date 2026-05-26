import numpy as np
import pywt
import cv2
from PIL import Image
from numba import njit
from skimage.restoration import estimate_sigma
from skimage.metrics import structural_similarity as ssim

class PixInsightMasterDenoisingPipeline:
    def __init__(self, master_light_path):
        """
        Etap I: Inicjalizacja potoku analitycznego.
        Wczytanie 32-bitowego liniowego pliku Master Light i normalizacja macierzy.
        """
        
       
        with Image.open(master_light_path) as img:
            
            self.master_light = np.array(img.convert('F'), dtype=np.float32)
            
        
        self.min_val = np.min(self.master_light)
        self.max_val = np.max(self.master_light)
        
        
        print(f"ADU: {self.min_val:.5f} to {self.max_val:.5f}")
    
        self.normalized_image = (self.master_light - self.min_val) / (self.max_val - self.min_val)
        
        
        self.sigma_noise = None
        self.luminance_mask = None
        self.denoised_output = None
        self.best_wavelet = 'db4'
        self.best_level = 3
       
    
