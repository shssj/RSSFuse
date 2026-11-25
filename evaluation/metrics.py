"""Six full-reference fusion measures used by the locked experiment (8-bit gray)."""
import math

import cv2
import numpy as np

from .protocol import uint8_image


def read_gray(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape {image.shape}: {path}")
    if gray.dtype != np.uint8:
        raise ValueError(f"Expected an 8-bit image, got {gray.dtype}: {path}")
    return gray


def entropy_metric(image):
    counts = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / image.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def standard_deviation(image):
    return float(np.std(image.astype(np.float64), ddof=0))


def mutual_information(first, second):
    indices = first.ravel().astype(np.int32) * 256 + second.ravel().astype(np.int32)
    joint = np.bincount(indices, minlength=256 * 256).reshape(256, 256)
    joint = joint.astype(np.float64) / first.size
    first_probability = joint.sum(axis=1)
    second_probability = joint.sum(axis=0)
    denominator = first_probability[:, None] * second_probability[None, :]
    valid = joint > 0
    return float(np.sum(joint[valid] * np.log2(joint[valid] / denominator[valid])))


def correlation(first, second):
    first = first.astype(np.float64)
    second = second.astype(np.float64)
    first -= first.mean()
    second -= second.mean()
    denominator = math.sqrt(float(np.sum(first * first) * np.sum(second * second)))
    if denominator == 0:
        return 0.0
    return float(np.sum(first * second) / denominator)


def gaussian_kernel(size, sigma):
    coordinate = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    xx, yy = np.meshgrid(coordinate, coordinate)
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel[kernel < np.finfo(np.float64).eps * kernel.max()] = 0
    return kernel / kernel.sum()


def valid_filter(image, kernel):
    # MATLAB filter2(kernel, image, 'valid') for the symmetric Gaussian kernel.
    filtered = cv2.filter2D(
        image, cv2.CV_64F, kernel, borderType=cv2.BORDER_CONSTANT
    )
    radius = kernel.shape[0] // 2
    if radius == 0:
        return filtered
    return filtered[radius:-radius, radius:-radius]


def vifp_mscale(reference, distorted):
    """Four-scale pixel-domain VIF, with sigma_nsq=2 as used in fusion code."""
    reference = reference.astype(np.float64)
    distorted = distorted.astype(np.float64)
    sigma_nsq = 2.0
    numerator = 0.0
    denominator = 0.0

    for scale in range(1, 5):
        size = 2 ** (4 - scale + 1) + 1
        kernel = gaussian_kernel(size, size / 5.0)

        if scale > 1:
            reference = valid_filter(reference, kernel)[::2, ::2]
            distorted = valid_filter(distorted, kernel)[::2, ::2]

        mean_reference = valid_filter(reference, kernel)
        mean_distorted = valid_filter(distorted, kernel)
        reference_variance = (
            valid_filter(reference * reference, kernel) - mean_reference * mean_reference
        )
        distorted_variance = (
            valid_filter(distorted * distorted, kernel) - mean_distorted * mean_distorted
        )
        covariance = (
            valid_filter(reference * distorted, kernel)
            - mean_reference * mean_distorted
        )

        reference_variance = np.maximum(reference_variance, 0.0)
        distorted_variance = np.maximum(distorted_variance, 0.0)
        gain = covariance / (reference_variance + 1e-10)
        residual_variance = distorted_variance - gain * covariance

        reference_flat = reference_variance < 1e-10
        gain[reference_flat] = 0.0
        residual_variance[reference_flat] = distorted_variance[reference_flat]
        reference_variance[reference_flat] = 0.0

        distorted_flat = distorted_variance < 1e-10
        gain[distorted_flat] = 0.0
        residual_variance[distorted_flat] = 0.0

        negative_gain = gain < 0
        residual_variance[negative_gain] = distorted_variance[negative_gain]
        gain[negative_gain] = 0.0
        residual_variance = np.maximum(residual_variance, 1e-10)

        numerator += float(
            np.sum(
                np.log10(
                    1.0
                    + gain * gain * reference_variance
                    / (residual_variance + sigma_nsq)
                )
            )
        )
        denominator += float(np.sum(np.log10(1.0 + reference_variance / sigma_nsq)))

    return numerator / denominator if denominator > 0 else 0.0


def evaluate(visible, infrared, fused):
    if visible.shape != infrared.shape or visible.shape != fused.shape:
        raise ValueError(
            f"Image sizes differ: VI={visible.shape}, IR={infrared.shape}, F={fused.shape}"
        )

    visible_float = visible.astype(np.float64)
    infrared_float = infrared.astype(np.float64)
    fused_float = fused.astype(np.float64)

    return {
        "SD": standard_deviation(fused),
        "MI": mutual_information(visible, fused)
        + mutual_information(infrared, fused),
        "EN": entropy_metric(fused),
        "CC": 0.5
        * (
            correlation(visible_float, fused_float)
            + correlation(infrared_float, fused_float)
        ),
        "SCD": correlation(fused_float - infrared_float, visible_float)
        + correlation(fused_float - visible_float, infrared_float),
        "VIF": vifp_mscale(visible_float, fused_float)
        + vifp_mscale(infrared_float, fused_float),
    }


def restoration_metrics(pred, target):
    """Channel-average valid 11x11 Gaussian SSIM; PSNR from one normalized RGB/L MSE."""
    if pred.shape != target.shape or min(pred.shape[-2:]) < 11:
        raise ValueError(f"Invalid restoration metric shapes: {pred.shape}, {target.shape}")
    x = pred.detach().clamp(0, 1).cpu().numpy().astype(np.float64)
    y = target.detach().cpu().numpy().astype(np.float64)
    mse = float(np.mean((x - y) ** 2))

    def blur(channel):
        return cv2.GaussianBlur(channel, (11, 11), 1.5)[5:-5, 5:-5]

    channel_ssim = []
    for xx, yy in zip(x, y):
        mux, muy = blur(xx), blur(yy)
        vx, vy = blur(xx * xx) - mux * mux, blur(yy * yy) - muy * muy
        cov = blur(xx * yy) - mux * muy
        score_map = ((2 * mux * muy + .01**2) * (2 * cov + .03**2)
                     / ((mux * mux + muy * muy + .01**2) * (vx + vy + .03**2)))
        channel_ssim.append(float(np.mean(score_map)))
    return -10 * math.log10(max(mse, 1e-12)), float(np.mean(channel_ssim))


def score(pred, batch, gray_vis, gray_ir):
    for name, target in (("vis", "vh"), ("ir", "ih"), ("fused", "vh")):
        if pred[name].shape != batch[target].shape or not pred[name].isfinite().all().item():
            raise ValueError(f"Invalid {name} prediction")
    fused_gray = cv2.cvtColor(uint8_image(pred["fused"]), cv2.COLOR_RGB2GRAY)
    values = evaluate(gray_vis, gray_ir, fused_gray)
    for channel, ref in (("vis", "vh"), ("ir", "ih")):
        values[channel + "_PSNR"], values[channel + "_SSIM"] = restoration_metrics(pred[channel], batch[ref])
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Non-finite metric")
    return values


