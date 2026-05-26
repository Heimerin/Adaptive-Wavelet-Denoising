# Adaptive-Wavelet-Denoising
Image processing Package for selective reconstruction of signal using multi-scale wavelet analysis (DWT) and Bayesian inference.


## Project Overview & Intent

In deep-sky astrophotography, photons collected from remote celestial objects are extremely faint, resulting in a low Signal-to-Noise Ratio (SNR). While stacking software (like PixInsight's WBPP) excels at calibrating and integrating sub-exposures, the integrated **Master Light** still suffers from residual stochastic noise.

This pipeline acts as an expert system that takes a 32-bit (or 16-bit) linear image from PixInsight, maps it into the **frequency-spatial domain using Discrete Wavelet Transform (DWT)**, and applies an adaptive **BayesShrink** thresholding algorithm. It intelligently differentiates between noise and structure, resulting in a cleaner background while preserving delicate nebular details and stellar profiles.

### Key Features
* **Hybrid Workflow:** Seamless integration with PixInsight/DSS linear data outputs.
* **Hyperparameter Optimization (Grid Search):** Automated selection of the optimal wavelet family and decomposition level based on structural similarity.
* **Luminance Masked Denoising:** Spatial modulation of the threshold via a custom luminance mask (stronger denoising in empty space, preservation in bright structures).
* **Iterative Residual Analysis:** A feedback loop that analyzes the difference map (rezydua) to prevent over-smoothing.

---

## Theoretical Background

The pipeline shifts image processing from the pixel domain into the multi-scale frequency domain, treating the image as a collection of localized spatial contrasts rather than static brightness points.

### 1. Mathematical Model of the Signal
The stacked Master Light $I(x,y)$ can be modeled as an additive mixture of the deterministic astrophysical signal and independent stochastic noise components:

$$I(x,y) = S(x,y) + \eta(x,y)$$

Where $S(x,y)$ is the pure signal and $\eta(x,y)$ represents the residual noise after stacking, modeled as a Gaussian distribution with zero mean and variance $\sigma^2$.

### 2. Discrete Wavelet Transform (DWT) Cascade
The image is passed through a system of low-pass ($h$) and high-pass ($g$) digital filters along rows and columns, splitting the matrix into four sub-bands for each decomposition level $j$:
* **$LL_j$ (Approximation):** Low-frequency components containing the core structural energy and global background gradients. This band is left untouched by the denoising core.
* **$HL_j$, $LH_j$, $HH_j$ (Details):** Horizontal, vertical, and diagonal high-frequency coefficients representing local contrasts and fine-grained noise.

### 3. BayesShrink & Soft Thresholding
The global residual noise variance $\sigma^2$ is calculated from the finest diagonal sub-band ($HH_1$) using Median Absolute Deviation (MAD), making it highly robust against stellar artifacts and hot pixels:

$$\sigma = \frac{\text{median}(|HH_1|)}{0.6745}$$

For each details sub-band, the pure signal variance $\sigma_x^2$ is isolated from total empirical variance $\sigma_y^2$:

$$\sigma_x = \sqrt{\max\left(0, \sigma_y^2 - \sigma^2\right)}$$

The data-driven **BayesShrink Threshold** $\lambda_B$ minimizes the Bayesian risk (mean squared error) for that specific scale:

$$\lambda_B = \frac{\sigma^2}{\sigma_x}$$

Finally, a **Soft Thresholding** operator shrinks the coefficients towards zero, ensuring smooth transitions without hard artifacts:

$$w_{\text{new}} = \text{sign}(w) \cdot \max(|w| - \lambda_B, 0)$$

## Architecture & Package Integration
The architecture distributes computing tasks across three major Python imaging libraries according to their native data analysis strengths:

[PixInsight 32-bit TIFF] ──> [Pillow (I/O & Normalization)]

[OpenCV (Luminance Mask)]

[PyWavelets (DWT + BayesShrink)] <── [scikit-image (Grid Search & SSIM)]

[Iterative Residual Loop]

[Exported clean 32-bit TIFF] ──> [PixInsight (or other astro-processing software) Non-linear Stretching]

## Package Roles
* **`opencv-python`**: Operates as a low-level engine for rapid pixel matrix manipulation. Used to extract structural luminance masks via large-kernel Gaussian blurs (`cv2.GaussianBlur`) to downscale lambda thresholds in high-signal regions.
* **`scikit-image`**: Serves as the validation oracle. Computes advanced metrics such as Structural Similarity Index (`ssim`) and Peak Signal-to-Noise Ratio for the tuning modules, and provides statistical verification tools via `estimate_sigma`.
* **`pillow`**: Manages bit-depth profiles (handling float32 spaces required by astronomy

## Validation (Results)
The performance of the pipeline is evaluated using objective statistical criteria:

* **Residual Map Analysis:** By tracking $$Residual = Original - Denoised$$, it is verified that the difference array consists purely of spatial white noise. If structural remnants are detected, the adaptive feedback system recalibrates the lambda modifiers.
* **SSIM/PSNR Maximization:** BayesShrink pipeline should show an improvement in the SSIM metric, therefore proving structural protection under severe streetching operations.
