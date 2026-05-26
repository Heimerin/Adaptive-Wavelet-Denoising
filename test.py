import os
from masterlight_read import PixInsightMasterDenoisingPipeline

if __name__ == "__main__":
    source_image = "/home/heimer/python_wavelet/Andromeda1.tif"
    
    if not os.path.exists(source_image):
        print(f"[Błąd] Nie znaleziono pliku: {source_image}")
    
  
    else:
        try:
           
            astro_pipeline = PixInsightMasterDenoisingPipeline(source_image)
            
        
            print(f"[Weryfikacja] Rozdzielczość załadowanego obrazu: {astro_pipeline.master_light.shape}")
            print(f"[Weryfikacja] Typ danych w macierzy NumPy: {astro_pipeline.master_light.dtype}")
            
        except Exception as e:
            print(f"[Błąd] Podczas otwierania pliku wystąpił błąd: {e}")