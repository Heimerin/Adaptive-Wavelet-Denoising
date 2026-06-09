import numpy as np
import pywt
import cv2
from PIL import Image
from numba import njit
from skimage.restoration import estimate_sigma
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import astropy as ap



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
        Etap I: Inicjalizacja potoku
        Automatyczne rozpoznawanie formatu (TIFF / FITS) 
        i wczytanie 32-bitowego liniowego pliku Master Light.
        """
        self.master_light_path = master_light_path
        
   
        _, ext = os.path.splitext(master_light_path)
        ext = ext.lower()
        
        if ext in ['.fit', '.fits']:
            print(f"[Inicjalizacja] Wykryto plik naukowy FITS: {master_light_path}")
            # Astropy otwiera plik jako listę bloków danych (HDU)
            with fits.open(master_light_path) as hdul:
                # Zazwyczaj dane obrazu znajdują się w głównym bloku (index 0)
                data = hdul[0].data
                if data is None:
                    data = hdul[1].data # Zabezpieczenie na wypadek innej struktury FITS
                
                self.master_light = np.array(data, dtype=np.float32)
                
        elif ext in ['.tif', '.tiff']:
            print(f"[Inicjalizacja] Wykryto plik TIFF: {master_light_path}")
            with Image.open(master_light_path) as img:
                self.master_light = np.array(img.convert('F'), dtype=np.float32)
                
        else:
            raise ValueError(f"Nieobsługiwany format pliku: {ext}. Użyj .tif lub .fit")
            
        self.min_val = np.min(self.master_light)
        self.max_val = np.max(self.master_light)
        
        print(f"Zakres sygnału (Min/Max): {self.min_val:.5f} do {self.max_val:.5f}")
    
        # Normalizacja statystyczna
        self.normalized_image = (self.master_light - self.min_val) / (self.max_val - self.min_val)
        
        self.sigma_noise = None
        self.luminance_mask = None
        self.denoised_output = None
        self.best_wavelet = 'db4'
        self.best_level = 3
        self.best_threshold_scale = 1.0
        self.metrics = {}

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
    
    def wavelet_denoising(self, wavelet=None, level=None, threshold_scale=1.0, store_result=True):
        """
        Bayes Shrink
        """

        if self.sigma_noise is None or self.luminance_mask is None:
            raise ValueError("Brak estymacji szumu i maski!")

        if wavelet is None:
            wavelet = self.best_wavelet
        if level is None:
            level = self.best_level

        #1 el rekompozycja obrazu 
        coeffs = pywt.wavedec2(self.normalized_image, wavelet, level=level)

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
                lambda_b = lambda_b * threshold_scale
                h_sub, w_sub= subband.shape
                resized_mask = cv2.resize(self.luminance_mask, (w_sub, h_sub), interpolation=cv2.INTER_AREA) #dodalem interpolacje, tutaj przy schodzeniu falkowym, zamiast probkowac punktowo, usredniamy ze zmniejszenego obszaru
                subband_filtered=f_adaptive_tresholding(subband, lambda_b, resized_mask)
                filtered_level.append(subband_filtered)

            new_coeffs.append(tuple(filtered_level))

        denoised_output = pywt.waverec2(new_coeffs, wavelet)
        h, w = self.normalized_image.shape
        denoised_output = denoised_output[:h, :w]
        denoised_output = np.clip(denoised_output, 0.0, 1.0)

        if store_result:
            self.denoised_output = denoised_output
            self.best_wavelet = wavelet
            self.best_level = level
            self.best_threshold_scale = threshold_scale

        return denoised_output
    
    # 5. GRID SEARCH 
    # -------------------------------------------------------------------------

    def optimize_hyperparameters(
        self,
        wavelets=("db2", "db4", "sym4", "coif2"),
        levels=(1, 2, 3),
        threshold_scales=(0.8, 1.0, 1.2),
    ):
        """
        Uwaga metodologiczna:
        ---------------------
        W prawdziwych danych astrofoto zwykle nie mamy obrazu idealnie czystego,
        więc nie da się policzyć klasycznego SSIM względem ground truth. Dlatego
        stosujemy kompromis:
        - SSIM(original, denoised) pilnuje, żeby nie niszczyć struktury obrazu;
        - std(residuals) premiuje usunięcie części szumu;
        - kara za średnią rezyduów ogranicza przesuwanie jasności tła.

        Wynik score nie jest absolutną miarą jakości, ale pomaga automatycznie
        wybrać rozsądne parametry startowe do prezentacji i testów.
        """
        if self.sigma_noise is None:
            self.estimate_residual_noise()
        if self.luminance_mask is None:
            self.create_luminance_mask()

        best_score = -np.inf
        best_result = None
        best_params = {}

        for wavelet in wavelets:
            for level in levels:
                for threshold_scale in threshold_scales:
                    try:
                        candidate = self.wavelet_denoising(
                            wavelet=wavelet,
                            level=level,
                            threshold_scale=threshold_scale,
                            store_result=False,
                        )

                        residuals = self.normalized_image - candidate

                        # SSIM bliski 1 oznacza, że struktura obrazu została zachowana.
                        structural_score = float(
                            ssim(
                                self.normalized_image,
                                candidate,
                                data_range=1.0,
                            )
                        )

                        # Im większe std rezyduów, tym więcej usunięto drobnej składowej.
                        # Nie chcemy jednak usuwać struktury, dlatego SSIM jest najważniejszy.
                        residual_std = float(np.std(residuals))
                        residual_mean_abs = float(abs(np.mean(residuals)))

                        # Heurystyczny score: zachowaj strukturę, usuń trochę szumu,
                        # nie przesuwaj globalnie jasności.
                        score = structural_score + 0.15 * residual_std - 0.50 * residual_mean_abs

                        print(
                            "Grid search:",
                            f"wavelet={wavelet}",
                            f"level={level}",
                            f"scale={threshold_scale}",
                            f"SSIM={structural_score:.5f}",
                            f"res_std={residual_std:.6f}",
                            f"score={score:.5f}",
                        )

                        if score > best_score:
                            best_score = score
                            best_result = candidate
                            best_params = {
                                "wavelet": wavelet,
                                "level": int(level),
                                "threshold_scale": float(threshold_scale),
                                "score": float(score),
                                "ssim_original_vs_denoised": structural_score,
                                "residual_std": residual_std,
                                "residual_mean_abs": residual_mean_abs,
                            }
                    except Exception as exc:
                        # Nie każda falka i poziom pasują do każdego rozmiaru obrazu.
                        print(f"[Grid search] Pominięto {wavelet}, level={level}, scale={threshold_scale}: {exc}")

        if best_result is None:
            raise RuntimeError("Grid search nie znalazł żadnej poprawnej konfiguracji.")

        self.best_wavelet = best_params["wavelet"]
        self.best_level = best_params["level"]
        self.best_threshold_scale = best_params["threshold_scale"]
        self.denoised_output = best_result
        self.metrics.update(best_params)

        print(
            f"[Grid search] Najlepsze parametry: wavelet={self.best_wavelet}, "
            f"level={self.best_level}, scale={self.best_threshold_scale}, "
            f"score={best_params['score']:.5f}"
        )
        return best_params
    
    #residual map Residual = original - denoised
    def export(self, output_path="master_light_denoised.tiff"):
        """
        Eksport do 32-bitowego pliku TIFF.
        Zapisujemy w znormalizowanym przedziale [0.0, 1.0], 
        aby zapewnić kompatybilność z programami takimi jak Darktable czy Siril.
        """
        if self.denoised_output is None:
            raise ValueError("Brak odszumionego obrazu!")
        
        print("[Eksport] Przygotowywanie 32-bitowego pliku TIFF (skala 0.0 - 1.0)...")
        
        # Bezpieczne skopiowanie znormalizowanego obrazu
        final_array = np.clip(self.denoised_output, 0.0, 1.0)

        # Zapis przez Pillow w formacie 32-bit float ('F')
        final_image = Image.fromarray(final_array.astype(np.float32), mode='F')
        final_image.save(output_path)
        print(f"[Sukces] Zapisano plik: {output_path}")

    def export_to_pixinsight(self, output_path="master_light_denoised.tiff"):
        self.export(output_path)


    def calculate_residuals(self, residual_path="master_light_residuals.tiff"):
        """
        Generowanie i zapis mapy usuniętego szumu w skali kompatybilnej.
        """
        if self.denoised_output is None:
            raise ValueError("Brak odszumionego obrazu!")
            
        # Obliczenie różnicy i przesunięcie szumu do średniej szarości (0.5)
        residuals = self.normalized_image - self.denoised_output
        residuals_shifted = np.clip(residuals + 0.5, 0.0, 1.0)
        
        # Zapis bezpośrednio w skali 0.0 - 1.0 z wymuszeniem trybu 'F'
        res_img = Image.fromarray(residuals_shifted.astype(np.float32), mode='F')
        res_img.save(residual_path)
        print(f"Mapa usuniętego szumu zapisana jako: {residual_path}")

    #======= Metryki
    def calculate_quality_metrics(self):
        """
        Obliczenie metryk jakości po odszumianiu.

        Metryki:
        --------
        MSE:
            Średni błąd kwadratowy między obrazem wejściowym i odszumionym.

        PSNR:
            Peak Signal-to-Noise Ratio liczony względem oryginału. Przy braku
            ground truth traktujemy go jako informację, jak mocno zmienił się obraz.

        SSIM:
            Strukturalne podobieństwo oryginału i wyniku. Blisko 1 oznacza, że
            geometria i struktura obrazu są dobrze zachowane.

        residual_*:
            Statystyki mapy różnicowej. Przydatne do omówienia w prezentacji.
        """
        residuals = self.normalized_image - self.denoised_output
        mse_value = float(np.mean((self.normalized_image - self.denoised_output) ** 2))

        metrics = {
            "best_wavelet": self.best_wavelet,
            "best_level": float(self.best_level),
            "threshold_scale": float(self.best_threshold_scale),
            "mse_original_vs_denoised": mse_value,
            "psnr_original_vs_denoised": float(psnr(self.normalized_image, self.denoised_output, data_range=1.0)),
            "ssim_original_vs_denoised": float(ssim(self.normalized_image, self.denoised_output, data_range=1.0)),
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "residual_min": float(np.min(residuals)),
            "residual_max": float(np.max(residuals)),
        }

        self.metrics.update(metrics)

        print("[Metryki] Wyniki jakości:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.8f}")
            else:
                print(f"  {key}: {value}")

        return self.metrics
    

    #============ Raport
    def save_report(self, report_path="master_light_report.txt"):
        if not self.metrics:
            print("[Raport] Brak metryk, liczę calculate_quality_metrics()...")
            self.calculate_quality_metrics()

        lines = [
            "Raport Adaptive Wavelet Denoising",
            "=================================",
            f"Plik wejściowy: {self.master_light_path}",
            f"Zakres ADU: {self.min_val:.8f} - {self.max_val:.8f}",
            f"Najlepsza falka: {self.best_wavelet}",
            f"Najlepszy poziom DWT: {self.best_level}",
            f"Skala progu: {self.best_threshold_scale}",
            "",
            "Metryki:",
        ]

        for key, value in sorted(self.metrics.items()):
            lines.append(f"- {key}: {value}")

        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write("\n".join(lines))

        print(f"[Raport] Zapisano raport jako: {report_path}")