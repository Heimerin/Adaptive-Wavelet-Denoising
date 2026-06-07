import numpy as np
import pywt
import cv2
from PIL import Image
from numba import njit
from skimage.restoration import estimate_sigma
from skimage.metrics import structural_similarity as ssim


#dorzucam njita do adaptywnego tresholdingu

@njit
def f_adaptive_tresholding(subband, lambda_b, mask):
    """
    Adaptacyjne progoowanie falkowe z modulacja maski do luminancji
    """
    height, width = subband.shape
    res = np.zeros_like(subband)

    for i in range(height):
        for j in range(width):
            val = subband[i,j]
            #prog bayesowski, ktory bedzie mnozony przez maske luminancji per piksel
            local_lambda = lambda_b * mask[i, j]

            if abs(val)  > local_lambda:
                #to sie odejmuje prog od sygnalu zachowujac znak
                res[i, j] = np.sign(val) * (abs(val) - local_lambda)
            else:
                res[i, j] = 0.0
    return res


class PixInsightMasterDenoisingPipeline:
    def __init__(self, master_light_path):
        """
        Etap I: Inicjalizacja potoku.
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

    def estimate_residual_noise(self):
        """
        Etap II: Falkowa estymacja szumu resztkowego Master Light.
        Wykorzystanie MAD oraz walidacji w scikit-image.
        """
        
        coeffs = pywt.wavedec2(self.normalized_image, 'db4', level=1)
       
        # coeffs[0] pasmo aproksymacji LL1
        # coeffs[1] krotka detali (LH1, HL1, HH1)
        i, level1_details = coeffs
        lh1, hl1, hh1 = level1_details

        median_absolute_deviation = np.median(np.abs(hh1))
        self.sigma_noise = median_absolute_deviation / 0.6745 #dla zbieżności odchylenia standardowego

        #estymacja szumu za pomocą scikit-image dla porównania
        scikit_sigma_value = estimate_sigma(self.normalized_image, channel_axis=None)

        print(f"(MAD): {self.sigma_noise:.6f}")
        print(f" Weryfikacja przez scikit-image: {scikit_sigma_value:.6f}")

        #obliczyć błąd względny:
        #quick search do duzych danych zeby zobacyc czy csa zaaleznosci pod katem opymaliuzacji gdzie ktora rodzina jest lepsza, zaczynajac od punktow startowych , zeby oproframowanie 
        #nie szukalo od zera zawsze, tykko robil quick search na sapisanej bazie i zoabczyc gdzie ktoere rodizny beda dominujace 

        rel_error = abs(self.sigma_noise - scikit_sigma_value) / scikit_sigma_value * 100
        print(f"Błąd względny między MAD a scikit-image: {rel_error:.2f}%")

    def create_luminance_mask(self, kernel_size=19):
        """
        ekstrakcja masek luminance i modulowanie odszumiania ~ sygnał
        """

        #kernel do rozmycia musi byc liczba nieparzysta, tak jest w splocie 
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        #wyzizilowanie niskich czestotliwości przez rozmycie gaussowskie
        low_freq = cv2.GaussianBlur(self.normalized_image, (kernel_size, kernel_size), 0)
        self.luminance_mask = 1.0 - low_freq
        print(f"Wymiary maski: {self.luminance_mask.shape}")
        print(f"Srednia waga tlumienia progu w poziomie cieni: {np.mean(self.luminance_mask):.4f}")
        

        return self.luminance_mask
    
    def wavelet_denoising(self):
        """
        Bayes Shrink
        """

        if self.sigma_noise is None or self.luminance_mask is None:
            raise ValueError("Brak estymacji szumu i maski!")

        #1 el rekompozycja obrazu 
        coeffs = pywt.wavedec2(self.normalized_image, self.best_wavelet, level=self.best_level)

        new_coeffs = [coeffs[0]]

        for level_idx in range(1, len(coeffs)):
            level_details = coeffs[level_idx]
            filtered_level=[]

            for subband in level_details:
                #wariancja calkowita dla subbandu detali
                var_y = np.var(subband)
                #wyizolowoanie wariancji szymu z estymacji
                sigma_x = np.sqrt(max(0, var_y - self.sigma_noise**2))
                if sigma_x ==0:
                    lambda_b = 1000.0 #duzy prog, ale celowo, zeby nie bylo dzielenia przez zero
                else:
                    lambda_b = (self.sigma_noise**2)/sigma_x
                h_sub, w_sub= subband.shape
                resized_mask = cv2.resize(self.luminance_mask, (w_sub, h_sub), interpolation=cv2.INTER_AREA) #dodalem interpolacje, tutaj przy schodzeniu falkowym, zamiast probkowac punktowo, usredniamy ze zmniejszenego obszaru
                subband_filtered=f_adaptive_tresholding(subband, lambda_b, resized_mask)
                filtered_level.append(subband_filtered)

            new_coeffs.append(tuple(filtered_level))

        self.denoised_output = pywt.waverec2(new_coeffs, self.best_wavelet)
        return self.denoised_output
    
    #residual map Residual = original - denoised
    def export(self, output_path = "master_light_denoised.tiff")
        """
        denormalizacja i zapis do 32 bit TIFF
        WAZNE: WRACAMY DO MORYGINALNEJ ROZPIETOSCI adu  Z MATRYCY KAMERy
        """

        if self.denoised_output is None:
            raise ValueError("Brak odszumionego obrazu!")
        
        #Na to zwrocic uwage:
        print(f"Przywracanie skali ADU ({self.min_val:.5f} do {self.max_val:.5f})...")
        final_array = self.denoised_output * (self.max_val - self.min_val) + self.min_val
        final_array = np.clip(final_array, self.min_val, self.max_val)

        #zapis oprzez pillow, musi byc bez kompresji do pozniejszej analizyw  astro soft
        final_image = Image.fromarray(final_array.astype(np.float32))
        final_image.save(output_path)


    def calculate_residuals(self, residual_path="master_light_residuals.tiff"):
        residuals = self.normalized_image - self.denoised_output
        residuals_shifted = residuals + 0.5
        residuals_shifted = np.clip(residuals_shifted, 0.0, 1.0)
        
        # Denormalizacja do ADU, aby plik był kompatybilny z naszym środowiskiem
        residuals_adu = residuals_shifted * (self.max_val - self.min_val) + self.min_val
        
        res_img = Image.fromarray(residuals_adu.astype(np.float32))
        res_img.save(residual_path)
        print(f"[Analityka] Mapa usuniętego szumu zapisana jako: {residual_path}")