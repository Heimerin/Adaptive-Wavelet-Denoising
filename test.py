from masterlight_read_featured_commented import PixInsightMasterDenoisingPipeline

if __name__ == "__main__":
    source_image = ""   #DO WPISANIA SCIEZKA DO ZDJECIA TIFF
    
    # Definicja plików wyjściowych
    output_clean = "master_light_denoised.tiff"
    output_resid = "master_light_residuals.tiff"
    output_report = "master_light_report.txt"
    
    # Etap I: Inicjalizacja
    pipeline = PixInsightMasterDenoisingPipeline(source_image)
    
    # Etap II: Estymacja szumu
    pipeline.estimate_residual_noise()
    
    # Etap III: Tworzenie maski
    pipeline.create_luminance_mask(kernel_size=19)
    
    # Etap IV: Główne odszumianie falkowe z Numba
    pipeline.wavelet_denoising()
    
    # Jeśli chcemy dobrać parametry automatycznie, można odkomentować:
    #pipeline.optimize_hyperparameters()
    
    # Etap V: Zapis wyników
    pipeline.export_to_pixinsight(output_clean)
    
    # Walidacja: Mapa różnicowa
    pipeline.calculate_residuals(output_resid)
    
    # Metryki i raport do prezentacji
    pipeline.calculate_quality_metrics()
    pipeline.save_report(output_report)
