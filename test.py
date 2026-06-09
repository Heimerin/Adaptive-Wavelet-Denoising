import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from masterlight_read import PixInsightMasterDenoisingPipeline

def generate_benchmark_plots(pipeline, output_prefix="benchmark_"):
    """
    Moduł wizualizacyjny do prezentacji wyników (Benchmarking).
    Porównuje nasz algorytm falkowy z klasycznymi filtrami przestrzennymi.
    """
    print("\n==================================================")
    print("GENEROWANIE WYKRESÓW PORÓWNAWCZYCH (BENCHMARK)")
    print("==================================================")
    
    img_orig = pipeline.normalized_image
    img_wavelet = pipeline.denoised_output
    
    print("[Bench] Obliczanie filtru Gaussa...")
    img_gauss = cv2.GaussianBlur(img_orig, (5, 5), 0)
    
    print("[Bench] Obliczanie filtru Medianowego...")
    # OpenCV wymaga float32 dla medianBlur
    img_median = cv2.medianBlur(img_orig.astype(np.float32), 3)

    # Obliczanie metryk
    metrics = {
        "Nasz Projekt (Falki)": {
            "SSIM": ssim(img_orig, img_wavelet, data_range=1.0),
            "PSNR": psnr(img_orig, img_wavelet, data_range=1.0)
        },
        "Rozmycie Gaussa": {
            "SSIM": ssim(img_orig, img_gauss, data_range=1.0),
            "PSNR": psnr(img_orig, img_gauss, data_range=1.0)
        },
        "Filtr Medianowy": {
            "SSIM": ssim(img_orig, img_median, data_range=1.0),
            "PSNR": psnr(img_orig, img_median, data_range=1.0)
        }
    }

    # Szukanie najjaśniejszej gwiazdy (jądra sygnału)
    _, _, _, max_loc = cv2.minMaxLoc(img_orig)
    star_x, star_y = max_loc
    
    # Wycięcie wąskiego paska wokół gwiazdy (np. 50 pikseli w lewo i prawo)
    radius = 50
    # Zabezpieczenie przed wyjściem poza obraz
    x_start, x_end = max(0, star_x - radius), min(img_orig.shape[1], star_x + radius)
    
    slice_orig = img_orig[star_y, x_start:x_end]
    slice_wavelet = img_wavelet[star_y, x_start:x_end]
    slice_gauss = img_gauss[star_y, x_start:x_end]
    slice_median = img_median[star_y, x_start:x_end]
    x_axis = np.arange(x_start, x_end)

    # ---------------------------------------------------------
    # WYKRES 1: Profil jasności (1D) Gwiazdy
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, slice_orig, label="Oryginał (Zaszumiony)", color='gray', alpha=0.5, linewidth=2)
    plt.plot(x_axis, slice_gauss, label="Rozmycie Gaussa", color='red', linestyle='--', linewidth=2)
    plt.plot(x_axis, slice_median, label="Filtr Medianowy", color='orange', linestyle='-.', linewidth=2)
    plt.plot(x_axis, slice_wavelet, label="Nasz Algorytm (Falki)", color='blue', linewidth=2)
    
    plt.title("Profil jasności pojedynczej gwiazdy (Porównanie zachowania krawędzi)")
    plt.xlabel("Pozycja piksela (Oś X)")
    plt.ylabel("Znormalizowana jasność ADU")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plot1_path = f"{output_prefix}1D_profile.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"[Sukces] Zapisano wykres profilu gwiazdy: {plot1_path}")
    plt.close()

    # ---------------------------------------------------------
    # WYKRES 2: Metryki SSIM
    # ---------------------------------------------------------
    labels = list(metrics.keys())
    ssim_values = [metrics[k]["SSIM"] for k in labels]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, ssim_values, color=['blue', 'red', 'orange'])
    
    # Dodanie wartości nad słupkami
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.title("Wskaźnik Podobieństwa Strukturalnego (SSIM)\nBliżej 1.0 = Mniej uszkodzeń struktury obrazu")
    plt.ylabel("Wartość SSIM")
    plt.ylim(0.5, 1.05) # Zawężenie osi Y by lepiej widzieć różnice
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    
    plot2_path = f"{output_prefix}SSIM_comparison.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"[Sukces] Zapisano wykres metryk SSIM: {plot2_path}")
    plt.close()


if __name__ == "__main__":
    # UWAGA: Wpisz poprawną ścieżkę do swojego pliku TIFF
    source_image = "/home/heimer/Pictures/La Palma/08_rho_50s1.tif"   
    
    output_clean = "master_light_denoised.tiff"
    output_resid = "master_light_residuals.tiff"
    output_report = "master_light_report.txt"
    
    if not os.path.exists(source_image):
        print(f"[Błąd] Nie znaleziono pliku: {source_image}")
    else:
        pipeline = PixInsightMasterDenoisingPipeline(source_image)
        pipeline.estimate_residual_noise()
        pipeline.create_luminance_mask(kernel_size=19)
        pipeline.wavelet_denoising()
        pipeline.export(output_clean)
        pipeline.calculate_residuals(output_resid)
        pipeline.calculate_quality_metrics()
        pipeline.save_report(output_report)
        
        # === NOWY KROK: Uruchomienie benchmarkingu ===
        generate_benchmark_plots(pipeline)