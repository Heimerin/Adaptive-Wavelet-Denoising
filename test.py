import os
from masterlight_read import PixInsightMasterDenoisingPipeline

if __name__ == "__main__":
    source_image = ""   #DO WPISANIA SCIEZKA DO ZDJECIA TIFF
    
    # Definicja plików wyjściowych
    output_clean = "/home/heimer/python_wavelet/Andromeda1_denoised.tif"
    output_resid = "/home/heimer/python_wavelet/Andromeda1_residuals.tif"
    
    if not os.path.exists(source_image):
        print(f"[Błąd] Nie znaleziono pliku: {source_image}")
    else:
        try:
         
          
            # Etap I: Inicjalizacja
            pipeline = PixInsightMasterDenoisingPipeline(source_image)
            
            # Etap II: Estymacja szumu
            pipeline.estimate_residual_noise()
            
            # Etap III: Tworzenie maski
            pipeline.create_luminance_mask(kernel_size=19)
            
            # Etap IV: Główne odszumianie falkowe z Numba
            pipeline.wavelet_denoising()
            
            # Etap V: Zapis wyników
            pipeline.export_to_pixinsight(output_clean)
            
            # Walidacja: Mapa różnicowa
            pipeline.calculate_residuals(output_resid)
            
        
            
        except Exception as e:
            print(f"[Błąd krytyczny] Wystąpił problem w potoku: {e}")