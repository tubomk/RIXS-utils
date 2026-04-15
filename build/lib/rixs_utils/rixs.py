from .helper_functions import print_analysis_parameters_template
from .helper_functions import median_filter_numpy
from .helper_functions import get_pgm_en
from .helper_functions import binned_spectrum
from .helper_functions import sum_spectra
from .helper_functions import make_mask, shear_and_crop_along_line, replace_lines_with_neighbor_mean
from .helper_functions import apply_custom_plot_style, apply_custom_plot_style_light
from .helper_functions import line, gaussian, voigt
from .helper_functions import parse_analysis_parameters
from numba import set_num_threads
import os
import sys
import copy
from copy import deepcopy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit, minimize_scalar
from scipy.ndimage import median_filter, distance_transform_edt, gaussian_filter1d
from scipy.interpolate import interp1d
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from matplotlib.axes import Axes
import math
from dataclasses import dataclass
phi = (1 + math.sqrt(5)) / 2


print("Test")

num_cpus = os.cpu_count()
set_num_threads(num_cpus - 2)  # type: ignore


cwd = Path.cwd()

scans_dir = cwd / "Scans"
spec_data_dir = cwd / "Spec_Files"
PathLike = Union[str, Path]


@dataclass(frozen=True)
# Container for filesystem and input-file configuration.
class RuntimeContext:
    cwd: Path
    scans_dir: Path
    spec_data_dir: Path
    analysis_parameters_path: Path
    dtype_ending: str
    file_dtype: Any
    has_metadata: bool


def _get_scan_dtype_ending(scans_folder: Path, verbose: bool = True) -> str:
    # Infer detector scan file extension from files in the scans folder.
    scan_x_files = sorted(scans_folder.glob("*x.*"))
    if scan_x_files:
        dtype_ending = scan_x_files[0].suffix.lstrip(".").lower()
        if verbose:
            print(
                f"Detected scan extension from Scans folder: .{dtype_ending}")
        return dtype_ending

    if verbose:
        print("No scan files found in Scans folder. Falling back to '.bin'.")
    return "bin"


def _get_scan_file_dtype(dtype_ending: str) -> Any:
    # Map file extension to the NumPy dtype used for detector files.
    if dtype_ending == "u16":
        return np.int16
    if dtype_ending == "bin":
        return np.float32
    raise ValueError(
        f"Unsupported scan extension '.{dtype_ending}'. Supported: .bin, .u16"
    )


def _resolve_optional_path(path_value: Optional[PathLike], default_path: Path) -> Path:
    # Resolve user-provided paths and keep relative paths anchored to the working directory.
    if path_value is None:
        return default_path

    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    return candidate


def _build_runtime_context(
    analysis_parameters_path: Optional[PathLike] = None,
    scans_path: Optional[PathLike] = None,
    spec_files_path: Optional[PathLike] = None,
    verbose: bool = True,
) -> RuntimeContext:
    # Validate required inputs and build shared runtime context.
    resolved_scans_dir = _resolve_optional_path(scans_path, scans_dir)
    resolved_spec_data_dir = _resolve_optional_path(
        spec_files_path, spec_data_dir)
    resolved_analysis_parameters_path = _resolve_optional_path(
        analysis_parameters_path, cwd / "Analysis_parameters.txt"
    )

    if verbose:
        print(
            f"Using Analysis Parameters file: {resolved_analysis_parameters_path}")
        print(f"Using Scans directory: {resolved_scans_dir}")
        print(f"Using Spec Files directory: {resolved_spec_data_dir}")

    if resolved_scans_dir.exists():
        if verbose:
            print("Scans folder found!")
    else:
        print(
            "Make sure all the Scans inclusive calibration Files "
            "(*x.bin, *y.bin Files) are in the Folder named 'Scans'!"
        )
        sys.exit()

    has_metadata = resolved_spec_data_dir.exists()
    if has_metadata:
        if verbose:
            print("Spec Files folder found!")
    else:
        print(
            "No folder with Spec Files found! Create a folder named Spec_Files (or whatever you specify) "
            "with the corresponding .spec file/s!"
        )

    dtype_ending = _get_scan_dtype_ending(resolved_scans_dir, verbose=verbose)
    file_dtype = _get_scan_file_dtype(dtype_ending)

    if resolved_analysis_parameters_path.exists():
        if verbose:
            print("Analysis parameters file found!")
    else:
        print(
            "Make sure to create the Analysis_parameters.txt file "
            "using the 'Write Analysis Parameter File.exe' executable,"
            " or use the template structure printed below to create it manually!"
        )
        print("\nExample structure for Analysis_parameters.txt:\n")
        print_analysis_parameters_template()
        sys.exit()

    return RuntimeContext(
        cwd=cwd,
        scans_dir=resolved_scans_dir,
        spec_data_dir=resolved_spec_data_dir,
        analysis_parameters_path=resolved_analysis_parameters_path,
        dtype_ending=dtype_ending,
        file_dtype=file_dtype,
        has_metadata=has_metadata,
    )


def _expand_scan_ranges(scan_ranges: Sequence[Tuple[int, int]]) -> List[int]:
    # Expand scan ranges into an explicit list of scan numbers.
    return [scan_num for start, end in scan_ranges for scan_num in range(start, end + 1)]


def _build_named_datasets(data_list: Sequence[dict], verbose: bool = True) -> List[Tuple[str, dict]]:
    # Build unique dataset names from shortDescription while preserving input order.
    total_counts: Dict[str, int] = {}
    for entry in data_list:
        base_name = str(entry["shortDescription"])
        total_counts[base_name] = total_counts.get(base_name, 0) + 1

    occurrence_counts: Dict[str, int] = {}
    named_datasets: List[Tuple[str, dict]] = []
    duplicate_mapping: Dict[str, List[str]] = {}

    for entry in data_list:
        base_name = str(entry["shortDescription"])
        occurrence_counts[base_name] = occurrence_counts.get(base_name, 0) + 1
        occurrence_index = occurrence_counts[base_name]

        if total_counts[base_name] == 1:
            data_name = base_name
        else:
            data_name = f"{base_name}_{occurrence_index}"
            if base_name not in duplicate_mapping:
                duplicate_mapping[base_name] = []
            duplicate_mapping[base_name].append(data_name)

        named_datasets.append((data_name, entry))

    if duplicate_mapping and verbose:
        print("Duplicate shortDescription detected. Using unique dataset names:")
        for base_name, resolved_names in duplicate_mapping.items():
            print(f"  {base_name} -> {', '.join(resolved_names)}")

    return named_datasets


def _collect_scan_pairs(
    scan_nums: Sequence[int], scans_folder: Path, dtype_ending: str
) -> List[Tuple[Path, Path, int]]:
    # Collect matching x/y detector files for the requested scan numbers.
    scan_num_set = set(scan_nums)
    return [
        (x_path, y_path, num)
        for x_path in sorted(scans_folder.glob(f"*x.{dtype_ending}"))
        for y_path in [x_path.with_name(x_path.stem[:-1] + f"y.{dtype_ending}")]
        if (num := int(x_path.stem[:-1].split("_")[-1])) in scan_num_set
    ]


def _run_iterative_outlier_removal(
    histogram: np.ndarray,
    threshold_range: np.ndarray,
    window_mad_filter: int,
    threshold_max: int,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    # Iteratively suppress outlier pixels based on local median statistics.
    total_steps = len(threshold_range)
    total_changed_pixels = 0

    histogram_sum = histogram.copy() + 1
    changed_mask = np.zeros_like(histogram_sum, dtype=bool)
    iteration = 0
    hist_grad = histogram_sum.copy()
    cleaned = histogram_sum.copy()

    if verbose:
        print(
            "Starting iterative outlier removal with thresholds: "
            f"{threshold_range[0]} down to {threshold_range[-1]}"
        )

    for step, threshold in enumerate(threshold_range, start=1):
        while True:
            median_img = median_filter_numpy(
                histogram_sum, size=window_mad_filter)
            mask_keep = histogram_sum - \
                median_img <= threshold * np.sqrt(median_img)
            cleaned = np.where(mask_keep, histogram_sum, median_img)

            changed_mask += ~mask_keep
            changed_pixels = int((~mask_keep).sum())
            total_changed_pixels += changed_pixels
            histogram_sum = cleaned.copy()
            iteration += 1

            if changed_pixels == 0:
                if threshold == threshold_max:
                    hist_grad = cleaned.copy()
                break

        if verbose:
            print(
                f"\rThreshold-Progress: {step}/{total_steps} | "
                f"Total Changed Pixels: {total_changed_pixels}",
                end="",
                flush=True,
            )

    cleaned = np.asarray(cleaned) - 1
    if verbose:
        print(
            f"\nTotal Changed Pixels after {iteration} iterations: {total_changed_pixels}"
        )
        print(
            f"Changed Pixels: {changed_mask.sum()} ({100 * changed_mask.mean():.2f}%)")

    return cleaned, changed_mask, hist_grad, total_changed_pixels, iteration


def calibration(
    show_plots=True,
    fig_size = 9,
    redo_calibration=True,
    save_parameters=True,
    verbose=True,
    analysis_parameters_path: Optional[PathLike] = None,
    scans_path: Optional[PathLike] = None,
    spec_files_path: Optional[PathLike] = None,
    colormap="gnuplot", test_run=False,
    calibration_parameters_path: Optional[PathLike] = None,
    calibration_parameters_output_dir: Optional[PathLike] = None,
):
    # Run detector calibration and save calibration parameters per dataset.

    runtime = _build_runtime_context(
        analysis_parameters_path=analysis_parameters_path,
        scans_path=scans_path,
        spec_files_path=spec_files_path,
        verbose=verbose,
    )

    calibration_parameters_dir = _resolve_optional_path(
        calibration_parameters_path, runtime.cwd
    )
    calibration_output_dir = _resolve_optional_path(
        calibration_parameters_output_dir, runtime.cwd
    )

    if verbose:
        print(
            f"Using calibration parameters lookup directory: {calibration_parameters_dir}"
        )
        print(
            f"Using calibration parameters output directory: {calibration_output_dir}")

    if save_parameters:
        calibration_output_dir.mkdir(parents=True, exist_ok=True)

    bad_pixels = []
    x_max, y_max = 4095, 4095

    data_list = parse_analysis_parameters(runtime.analysis_parameters_path)
    named_datasets = _build_named_datasets(data_list, verbose=verbose)
    total_data_sets = len(named_datasets)
    successfully_calibrated_datasets = 0

    # Correct Calibration and Scan Energies
    for _, dataset in named_datasets:
        calib_scan_nums = np.arange(
            dataset['calibrationScanNums'][0][0], dataset['calibrationScanNums'][0][1] + 1)
        scan_nums = np.arange(dataset['scanNums']
                              [0][0], dataset['scanNums'][0][1] + 1)
        if runtime.has_metadata:
            dataset['calibrationEnergies'] = get_pgm_en(
                calib_scan_nums, spec_dir=runtime.spec_data_dir)
            dataset['incidentEnergy'] = np.mean(
                np.array(get_pgm_en(scan_nums, spec_dir=runtime.spec_data_dir)))

    #### #### #### #### #### ####

    for data_name, dataset in named_datasets:
        if (not runtime.has_metadata) and (
            dataset.get("calibrationEnergies") is None
            or len(dataset.get("calibrationEnergies", [])) == 0
        ):
            print(
                f"Warning: Cannot run calibration for {data_name}. "
                "No Spec_Files metadata found and no calibrationEnergies provided "
                "in Analysis_parameters."
            )
            continue

        if not redo_calibration:
            calibration_file = calibration_parameters_dir / \
                f"calibration_parameters_{data_name}.npy"
            if calibration_file.exists():
                (m_fit, b_fit, y_max, max_sigma_x, max_sigma_y,
                 mCalibration, bCalibration, var_m, var_b, cov_mb) = np.load(
                    calibration_file
                )
                if verbose:
                    print(f"{calibration_file.name} already exists in {calibration_parameters_dir}, Parameters loaded.\n"
                          "Skip the calibration")
                successfully_calibrated_datasets += 1
                continue
            else:
                calibration_Done = False
                if verbose:
                    print(f"{calibration_file.name} does NOT exist in {calibration_parameters_dir}.\n"
                          "Running calibration routine")

        # Retrieve calibration scan files
        calibrationScanRanges = dataset["calibrationScanNums"]
        calibrationFileList = _expand_scan_ranges(calibrationScanRanges)

        calibrationEnergies = dataset["calibrationEnergies"]
        numCalibFiles = len(calibrationFileList)
        x_max, y_max = 4095, 4095  # Detector Size

        # Estimate Monochromator uncertainty in Pixel
        deltaE_mono = 0.003
        est_energy_window = (np.max(dataset['calibrationEnergies']) -
                             np.min(dataset['calibrationEnergies']))
        est_deltaE_mono = np.ceil(deltaE_mono/(est_energy_window/x_max))

        files = [
            [x, y, num]
            for x in sorted(runtime.scans_dir.glob(f"*x.{runtime.dtype_ending}"))
            for y in [x.with_name(x.stem[:-1] + f"y.{runtime.dtype_ending}")]
            if (num := int(x.stem[:-1].split("_")[-1])) in calibrationFileList
        ]

        calibrationFiles = []
        for x_path, y_path, num in files:
            eventXY = np.vstack(
                (
                    # type: ignore
                    np.fromfile(x_path, dtype=runtime.file_dtype),
                    # type: ignore
                    np.fromfile(y_path, dtype=runtime.file_dtype),
                )
            ).T
            mask = (eventXY[:, 0] <= x_max) & (eventXY[:, 1] <= y_max)
            eventXY = eventXY[mask].astype(np.int16)
            calibrationFiles.append((num, eventXY))

        # break

        # Create 2D histograms from calibration data
        allHistograms = []
        x_bins, y_bins = np.arange(x_max + 1), np.arange(y_max + 1)
        for i in range(len(calibrationFiles)):
            x, y = calibrationFiles[i][1].T
            histogram, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
            for bad_pixel in bad_pixels:
                histogram[bad_pixel] = 0
            allHistograms.append(histogram)

        # Find y_max from summed histogram
        histSum = np.sum(allHistograms, axis=0)
        counts_per_y = histSum.sum(axis=0)
        nonzero = np.nonzero(counts_per_y)[0]
        y_max = nonzero.max() + 1 if nonzero.size > 0 else y_max

        # Trim histograms
        for i, hist in enumerate(allHistograms):
            allHistograms[i] = hist[:, :y_max]
        histSum = np.sum(allHistograms, axis=0)

        # 1D Gaussian fit to each horizontal and vertical profile
        peak_centers = np.zeros((numCalibFiles, 2))
        stdDevs = np.zeros((numCalibFiles, 2))
        mu_err_h_all = np.zeros(numCalibFiles)

        for i, hist in enumerate(allHistograms):
            (x_h, y_h) = (np.arange(hist.shape[0]), hist.sum(axis=1))
            (x_v, y_v) = (np.arange(hist.shape[1]), hist.sum(axis=0))

            # Robust initial parameters
            amplitude_h = y_h.max() - y_h.min()
            amplitude_v = y_v.max() - y_v.min()

            center_h = np.sum(x_h * y_h) / np.sum(y_h)
            center_v = np.sum(x_v * y_v) / np.sum(y_v)

            sigma_h = np.sqrt(np.sum(y_h * (x_h - center_h)**2) / np.sum(y_h))
            sigma_v = np.sqrt(np.sum(y_v * (x_v - center_v)**2) / np.sum(y_v))

            offset_h = y_h.min()
            offset_v = y_v.min()

            p0_h = [amplitude_h, center_h, sigma_h, offset_h]
            p0_v = [amplitude_v, center_v, sigma_v, offset_v]

            # Poisson-Errors for Profile
            yerr_h = np.sqrt(np.maximum(y_h, 1.0))
            yerr_v = np.sqrt(np.maximum(y_v, 1.0))

            params_h, pcov_h = curve_fit(gaussian, x_h, y_h, p0=p0_h, sigma=yerr_h,
                                         absolute_sigma=True)
            params_v, pcov_v = curve_fit(gaussian, x_v, y_v, p0=p0_v, sigma=yerr_v,
                                         absolute_sigma=True)

            muFit_h, sigmaFit_h = params_h[1], abs(params_h[2])
            muFit_v, sigmaFit_v = params_v[1], abs(params_v[2])

            peak_centers[i] = muFit_h, muFit_v
            stdDevs[i] = sigmaFit_h, sigmaFit_v

            mu_err_h = float(np.sqrt(pcov_h[1, 1])
                             ) if pcov_h is not None else np.nan
            mu_err_h_all[i] = mu_err_h if np.isfinite(mu_err_h) else np.nan

        x_line = np.linspace(0, x_max - 1, x_max - 1)
        p0_l = [1, 1]
        if data_name in ['Sample_31_8K', 'Sample_30_8K', 'Sample_16_8K']:
            params_l, _ = curve_fit(
                line, peak_centers[:-2, 0], peak_centers[:-2, 1], p0=p0_l)
            if verbose:
                print(f"Exclude last elastic peaks for Measurement {data_name}, due to cut-off \n"
                      "(Calibration not affected)")
        else:
            params_l, _ = curve_fit(
                line, peak_centers[:, 0], peak_centers[:, 1], p0=p0_l)
        m_fit, b_fit = params_l

        # Perform calibration only with the x-positions of the elastic peaks.

        max_sigma_x, max_sigma_y = np.max(stdDevs[:, 0]), np.max(stdDevs[:, 1])

        # Linear fit: Energy = m * Pixel + b
        p0_l = [1, 0]
        x_meas = peak_centers[:, 0]
        y_meas = np.asarray(calibrationEnergies, dtype=float)
        sx = mu_err_h_all.copy()
        if not np.all(np.isfinite(sx)):
            # Fallback: use the median of valid uncertainties.
            med = np.nanmedian(sx)
            sx[~np.isfinite(sx)] = med if np.isfinite(med) else 1.0
        # Same approach as above: weighted curve_fit with absolute_sigma=True
        # Combine monochromator y-uncertainty and propagated x-uncertainty (m*sx)
        m_guess = 1.0
        b_guess = 0.0
        for _ in range(3):
            sy_eff = np.sqrt(deltaE_mono**2 + (m_guess * sx)**2)
            params_cal, pcov_cal = curve_fit(
                line, x_meas, y_meas, p0=[m_guess, b_guess],
                sigma=sy_eff, absolute_sigma=True
            )
            m_guess, b_guess = params_cal

        mCalibration, bCalibration = params_cal  # type: ignore

        var_m = pcov_cal[0, 0]  # type: ignore
        var_b = pcov_cal[1, 1]  # type: ignore
        cov_mb = pcov_cal[0, 1]  # type: ignore

        # params_l, pcoval_l = curve_fit(line, peak_centers[:, 0], calibrationEnergies, p0=p0_l)
        # mCalibration, bCalibration = params_l
        # var_m, var_b, cov_mb = pcoval_l[0,0], pcoval_l[1,1], pcoval_l[0,1]

        if show_plots == True:
            letters = list("abcdefghijklmnopqrstuvwxyz")
            titleFontsize = 18
            subtitleFontsize = 16

            figure_1, axes_1 = plt.subplots(
                ncols=1, nrows=3, figsize=(fig_size, 1.5*(fig_size/phi)), gridspec_kw={'height_ratios': [2, 2, 2]}
            )

            figure_1.suptitle(data_name,
                              fontsize=titleFontsize)
            mask = make_mask(histSum.T.shape, (m_fit, b_fit), max_sigma_y, 5.8)
            histSum = histSum * mask.T
            y_indices_with_true = np.where(mask.any(axis=1))[0]
            min_y_mask = y_indices_with_true.min()
            max_y_mask = y_indices_with_true.max()

            k_ = min(100, histSum.size)
            mean_bottomk = np.mean(np.partition(
                histSum.ravel(), k_ - 1)[:k_])
            mean_topk = np.mean(np.partition(histSum.ravel(), -k_)[-k_:])
            vmin = int(mean_bottomk + 1)
            vmax = int(mean_topk + 1)

            axes_1[0].imshow(
                histSum.T,
                # norm=LogNorm(vmin=1, vmax=histSum.max()),
                vmin=vmin, vmax=vmax,
                cmap=colormap,
                # cmap=plt.colormaps()[1],
                aspect="auto",
                origin="lower",
            )

            axes_1[0].scatter(
                peak_centers[:, 0], peak_centers[:, 1], marker="+", s=90, c="white", zorder=2
            )
            axes_1[0].set_ylim(min_y_mask, max_y_mask)
            axes_1[0].set_xlabel("Pixel")
            axes_1[0].set_ylabel("Pixel")
            axes_1[0].set_title("Elastic Lines - Calibration", fontsize=16)

            axes_1[1].scatter(
                peak_centers[:, 0], calibrationEnergies, marker="x", facecolor="white", s=60, zorder=3
            )
            axes_1[1].plot(
                x_line, line(x_line, mCalibration, bCalibration), c="magenta", lw=2, zorder=0
            )
            axes_1[1].set_title(f"Calibration function", fontsize=16)
            axes_1[1].set_xlabel("Pixel")
            axes_1[1].set_ylabel("Energy (eV)")
            axes_1[1].set_xlim(0, x_max + 1)
            axes_1[1].annotate(
                rf"$E = m \cdot \mathrm{{Pixel}} + b$" + "\n" +
                rf"$m = {mCalibration:.7f}, \; b = {bCalibration:.5f}$ ",
                xy=(0.02, 0.96),
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=14,
            )

            fwhm_constant = 2 * np.sqrt(2 * np.log(2))
            axes_1[2].set_title(f"Resolution", fontsize=16)
            axes_1[2].scatter(peak_centers[:, 0]*mCalibration + bCalibration,
                              stdDevs[:, 0]*mCalibration*fwhm_constant, marker="x", facecolor="white", s=60,)
            axes_1[2].set_xlabel('Energy (eV)')
            axes_1[2].set_ylabel(
                r"$2\sqrt{2\ln(2)}\,\sigma\;\mathrm{(eV)}$", fontsize=12)

            plt.tight_layout()
            plt.show()

        if not show_plots and verbose:
            print(f"Line Parameters Calibration for Data {data_name} (Energy = m*Pixel + b):"
                  f"\n m = {mCalibration:.7f} \n b = {bCalibration:.5f} \n")

        calibration_parameters = np.array(
            [m_fit, b_fit, y_max, max_sigma_x, max_sigma_y, mCalibration, bCalibration, var_m, var_b, cov_mb])
        std_devs = np.stack((peak_centers[:, 0]*mCalibration + bCalibration,
                            stdDevs[:, 0]*mCalibration), axis=1)
        if save_parameters:
            np.save(
                calibration_output_dir / f"calibration_parameters_{data_name}.npy", calibration_parameters)
            np.save(calibration_output_dir /
                    f"std_devs_{data_name}.npy", std_devs)

        successfully_calibrated_datasets += 1

        if test_run:
            break

    print(
        f"Successfully completed calibration for {successfully_calibrated_datasets}/{total_data_sets} data set(s)."
    )


####    ####    ####    #####

def process_RIXS(
    show_plots=True,
    fig_size=9,
    dark_mode=True,
    e_IN_correction=True,
    dead_and_hyperactive_line_correction=True,
    threshold_filtering=6.0,
    vertical_tolerance=3.5,
    verbose=True,
    analysis_parameters_path: Optional[PathLike] = None,
    scans_path: Optional[PathLike] = None,
    spec_files_path: Optional[PathLike] = None,
    calibration_parameters_path: Optional[PathLike] = None,
    test_run=False,
    colormap="gnuplot",
    bin_size_plot=8
):
    # Process raw RIXS detector scans into calibrated 1D/2D spectra.

    runtime = _build_runtime_context(
        analysis_parameters_path=analysis_parameters_path,
        scans_path=scans_path,
        spec_files_path=spec_files_path,
        verbose=verbose,
    )

    calibration_parameters_dir = _resolve_optional_path(
        calibration_parameters_path, runtime.cwd
    )
    if verbose:
        print(
            f"Using calibration parameters lookup directory: {calibration_parameters_dir}"
        )

    bad_pixels = []
    x_max, y_max = 4095, 4095

    data_list = parse_analysis_parameters(runtime.analysis_parameters_path)
    named_datasets = _build_named_datasets(data_list, verbose=verbose)
    total_data_sets = len(named_datasets)

    zoom_multiplier = 1
    if dark_mode:
        apply_custom_plot_style()
    else:
        apply_custom_plot_style_light()

    x_max_bins, y_max_bins = 4095, 4095

    window_mad_filter = 9
    threshold_max = 30

    threshold_range = np.arange(
        threshold_max, int(np.ceil(threshold_filtering)), -1)
    if threshold_filtering not in threshold_range:
        threshold_range = np.append(threshold_range, threshold_filtering)

    threshold_dead_hyperactiveline = 0.4
    filter_size_dead_hyperactive = 7

    processed_data = {}
    successfully_processed_datasets = 0

    for index, (data_name, dataset) in enumerate(named_datasets, start=1):

        processed_data[data_name] = {}
        calibration_file = True
        sigma_mean = None

        if verbose:
            print(
                f"Processing data set {index}/{total_data_sets}: {data_name}")
        try:
            calibration_file_path = calibration_parameters_dir / \
                f"calibration_parameters_{data_name}.npy"
            (m_fit, b_fit, y_max_bins, _, max_sigma_y,
                mCalibration, bCalibration, _, _, _) = np.load(
                    calibration_file_path
            )
            std_devs_path = calibration_parameters_dir / \
                f"std_devs_{data_name}.npy"
            if std_devs_path.exists():
                std_devs = np.load(std_devs_path)
                sigma_mean = float(np.mean(std_devs[:, 1]))
        except FileNotFoundError:
            print(
                f"Calibration parameters for {data_name} not found in {calibration_parameters_dir}.")
            calibration_file = False
            m_fit, b_fit, max_sigma_y = 1, 0, 4095
            mCalibration, bCalibration = 1, 0

        apply_e_in_correction = e_IN_correction and calibration_file
        deviation_idx = {}

        ScanRanges = dataset["scanNums"]
        scan_nums = _expand_scan_ranges(ScanRanges)

        pgm_values = get_pgm_en(scan_nums, spec_dir=runtime.spec_data_dir)
        valid_pgm_values = np.array(
            [value for value in pgm_values if value is not None], dtype=float
        )

        if valid_pgm_values.size > 0:
            mean_E_In = float(np.mean(valid_pgm_values))
            E_in = np.array(
                [value if value is not None else mean_E_In for value in pgm_values],
                dtype=float,
            )
        else:
            incident_energy = dataset.get("incidentEnergy")
            if incident_energy is not None and np.isfinite(float(incident_energy)):
                mean_E_In = float(incident_energy)
            else:
                mean_E_In = 0.0

            E_in = np.full(len(scan_nums), mean_E_In, dtype=float)
            if apply_e_in_correction:
                print(
                    f"No valid spec metadata found for {data_name}. "
                    "Disabling e_IN correction for this dataset."
                )
                apply_e_in_correction = False

        # Calculate deviation in E_in and corresponding pixel shift
        if apply_e_in_correction:
            deviation = np.round((mean_E_In - E_in) /
                                 mCalibration).astype(int)  # type: ignore
            deviation_idx = {val: np.where(deviation == val)[
                0].tolist() for val in np.unique(deviation)}

        processed_data[data_name]['scan_nums'] = len(scan_nums)
        files = _collect_scan_pairs(
            scan_nums=scan_nums,
            scans_folder=runtime.scans_dir,
            dtype_ending=runtime.dtype_ending,
        )

        all_events = []

        # Dead Line for Sample_16_8K is at y=1, so start with y=2 to exclude it from the histogram and later processing steps
        if data_name in ['Sample_31_RT',
                         'Sample_30_RT',
                         'Sample_16_RT',
                         'Sample_31_8K',
                         'Sample_30_8K',
                         'Sample_16_8K']:
            x_bins, y_bins = np.arange(
                x_max_bins + 1), np.arange(2, y_max_bins + 1)
        else:
            x_bins, y_bins = np.arange(
                x_max_bins + 1), np.arange(y_max_bins + 1)
        for i, (x_path, y_path, num) in enumerate(files):
            eventXY = np.vstack(
                (np.fromfile(x_path, dtype=runtime.file_dtype),
                 np.fromfile(y_path, dtype=runtime.file_dtype),)).T
            eventXY = eventXY.astype(np.int16)
            all_events.append(eventXY)

        all_events = np.vstack(all_events)

        y_vals = all_events[:, 1]
        if y_vals.size > 0:
            y_max_bins = int(y_vals.max()) + 1
        y_bins = np.arange(2, y_max_bins + 1)

        hist_counts, _, _ = np.histogram2d(
            all_events[:, 0], all_events[:, 1],
            bins=[x_bins, y_bins])

        cleaned, changed_mask, _, total_changed_pixels, iteration = _run_iterative_outlier_removal(
            histogram=hist_counts,
            threshold_range=threshold_range,
            window_mad_filter=window_mad_filter,
            threshold_max=threshold_max,
            verbose=verbose,
        )
        histogram_sum = cleaned.copy()
        if verbose:
            print(
                f"\nTotal Changed Pixels after {iteration} iterations: {total_changed_pixels}")
            print(
                f"Changed Pixels: {changed_mask.sum()} ({100*changed_mask.mean():.2f}%)")

    #### #### DEAD LINE AND HYPERACTIVE LINE DETECTION #### ####

        #### Search for dead lines ###
        x_profile = cleaned.sum(axis=1)  # type: ignore
        half_size = filter_size_dead_hyperactive // 2
        local_median = median_filter(
            x_profile, size=filter_size_dead_hyperactive)
        ratio = x_profile / (local_median + 1e-9)

        dead_lines = np.where(ratio < 1 - threshold_dead_hyperactiveline)[0]
        hyperactive_lines = np.where(
            ratio > 1 + threshold_dead_hyperactiveline)[0]

        # x_bins, y_bins = np.arange(x_max + 1), np.arange(y_max + 1)
        hist_total = np.zeros_like(cleaned)

        if verbose:
            if apply_e_in_correction:
                print(
                    f"Total deviations detected from mean Incoming Energies: {len(list(deviation_idx.keys()))-1} ")
        if apply_e_in_correction:
            for deviation_value in list(deviation_idx.keys()):
                idx_list = deviation_idx[deviation_value]
                all_events = []
                for idx in idx_list:
                    x_path, y_path, _ = files[idx]
                    eventXY = np.vstack(
                        (np.fromfile(x_path, dtype=runtime.file_dtype),
                         np.fromfile(y_path, dtype=runtime.file_dtype),)).T
                    mask = (0 <= eventXY[:, 0]) & (
                        eventXY[:, 0] <= x_max) & (eventXY[:, 1] <= y_max)
                    eventXY = eventXY[mask].astype(np.int16)
                    all_events.append(eventXY)
                all_events = np.vstack(all_events)
                hist_temp, _, _ = np.histogram2d(
                    all_events[:, 0], all_events[:, 1], bins=[x_bins, y_bins])
                # hist_temp = shear_and_crop_along_line(
                #     hist_temp, m_fit, b_fit, max_sigma_y, width_factor=vertical_tolerance
                # )
                hist_temp = np.where(~changed_mask, hist_temp, 0)
                if deviation_value != 0:
                    shifted = np.zeros_like(hist_temp)
                    if deviation_value > 0:
                        shifted[deviation_value:] = hist_temp[:-deviation_value]
                    else:
                        shifted[:deviation_value] = hist_temp[-deviation_value:]
                    hist_total += shifted
                else:
                    hist_total += hist_temp
        else:
            hist_total = np.where(~changed_mask, histogram_sum, 0)
            # hist_total = cleaned

        x_profile = np.sum(hist_total, axis=1)
        half_size = filter_size_dead_hyperactive // 2
        local_median = median_filter(
            x_profile, size=filter_size_dead_hyperactive)
        ratio = x_profile / (local_median + 1e-9)

        dead_lines = np.where(ratio < 1 - threshold_dead_hyperactiveline)[0]
        hyperactive_lines = np.where(
            ratio > 1 + threshold_dead_hyperactiveline)[0]
        if dead_and_hyperactive_line_correction:
            hist_total = replace_lines_with_neighbor_mean(
                hist_total, dead_lines, filter_size_dead_hyperactive, 'dead', verbose=verbose)

        ### Shear and Crop ###
        if calibration_file:
            hist_total = shear_and_crop_along_line(
                hist_total, m_fit, b_fit, max_sigma_y, width_factor=15
            )
            hist_counts = shear_and_crop_along_line(
                hist_counts, m_fit, b_fit, max_sigma_y, width_factor=15
            )
            changed_mask = shear_and_crop_along_line(
                changed_mask.astype(int), m_fit, b_fit, max_sigma_y, width_factor=15
            ).astype(bool)

        ### Compute Vertical Profile to constrain signal region ###
        hist_total = np.asarray(hist_total)

        vertical_profile = (
            np.arange(hist_total.shape[1]), np.sum(hist_total, axis=0))
        amplitude_guess = vertical_profile[1].max() - vertical_profile[1].min()
        center_guess = np.sum(
            vertical_profile[0] * vertical_profile[1]) / np.sum(vertical_profile[1])
        sigma_guess = np.sqrt(np.sum(
            vertical_profile[1] * (vertical_profile[0] - center_guess)**2) / np.sum(vertical_profile[1]))
        offset_guess = vertical_profile[1].min()

        p0 = [amplitude_guess, center_guess, sigma_guess, offset_guess]

        # Compute Gaussian Fit to Vertical Profile
        params_gauss, pcov_gauss = curve_fit(
            gaussian, vertical_profile[0], vertical_profile[1], p0=p0)
        _, center_fit, sigma_fit, _ = params_gauss

        # Clamp crop window to valid array bounds and avoid empty slices.
        y_size = hist_total.shape[1]
        y_min_hist = int(np.floor(center_fit - sigma_fit * vertical_tolerance))
        y_max_hist = int(np.ceil(center_fit + sigma_fit * vertical_tolerance))

        y_min_hist = max(0, y_min_hist)
        y_max_hist = min(y_size, y_max_hist)

        if y_max_hist <= y_min_hist:
            center_idx = int(np.clip(round(center_fit), 0, y_size - 1))
            y_min_hist = max(0, center_idx - 1)
            y_max_hist = min(y_size, center_idx + 2)

        hist_total_2 = hist_total[:, y_min_hist:y_max_hist]

        processed_data[data_name]["E_in"] = mean_E_In
        processed_data[data_name]["histogram"] = hist_total_2
        if calibration_file and sigma_mean is not None:
            processed_data[data_name]["resolution"] = sigma_mean
        if apply_e_in_correction and calibration_file:
            energy_axis = (
                mean_E_In - (np.arange(hist_total_2.shape[0])*mCalibration + bCalibration))[::-1]
            processed_data[data_name]['x'] = energy_axis
            processed_data[data_name]['Loss_Scale'] = True
            processed_data[data_name]['y'] = np.sum(hist_total_2, axis=1)[::-1]
            if verbose:
                print(f"Applied E_in deviation correction. Storing as loss spectrum!")

        elif calibration_file:
            energy_axis = np.arange(
                hist_total_2.shape[0])*mCalibration + bCalibration
            processed_data[data_name]['x'] = energy_axis
            processed_data[data_name]['Loss_Scale'] = False
            if verbose:
                print(
                    f"No E_in deviation correction applied. Storing as direct spectrum!")
            processed_data[data_name]['y'] = np.sum(hist_total_2, axis=1)

        else:
            processed_data[data_name]['x'] = np.arange(hist_total_2.shape[0])
            processed_data[data_name]['Loss_Scale'] = False
            print(
                f"No calibration file found. Storing spectrum with pixel axis and no loss scale!")
            processed_data[data_name]['y'] = np.sum(hist_total_2, axis=1)

        if verbose:
            print(
                f"Finished processing data set {index}/{total_data_sets}: {data_name}\n")
            print("-"*80)

    ### Plotting after outlier removal and line correction ###
        if show_plots:
            layout = [
                ["1", "3"],
                [None, "5"],
                ["2",  "4"],
            ]
            fig_spectra, axs = plt.subplot_mosaic(layout, figsize=(
                zoom_multiplier*fig_size, zoom_multiplier*fig_size/phi), empty_sentinel=None)  # type:ignore
            fig_spectra.suptitle(f"{data_name}", fontsize=16)
            axs: Dict[str, Axes] = axs

            axs["2"].sharex(axs['1'])
            axs["3"].sharex(axs["1"])
            axs["3"].sharey(axs["1"])
            axs["4"].sharex(axs["1"])
            axs["5"].sharex(axs["1"])
            axs["5"].sharey(axs["1"])  # axs["4"].sharey(axs["2"])

            axs["1"].tick_params(labelbottom=False)
            axs["3"].tick_params(labelbottom=False)
            axs["5"].tick_params(labelbottom=False)
            axs["3"].tick_params(labelleft=False)
            axs["4"].tick_params(labelleft=False)
            axs["5"].tick_params(labelleft=False)

            axs["1"].annotate('Raw Spectrum',          xy=(0.02, 0.94), xycoords="axes fraction",
                              ha="left", va="top", fontsize=12, color='w',
                              bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.95))
            axs["2"].annotate('Raw Spectrum 1D (Cnt. Norm.)', xy=(0.02, 0.94),
                              xycoords="axes fraction", ha="left", va="top", fontsize=12)
            axs["3"].annotate('Cleaned Spectrum', xy=(0.02, 0.94), xycoords="axes fraction",
                              ha="left", va="top", fontsize=12, color='w',
                              bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.95))
            axs["4"].annotate('Cleaned Spectrum 1D (Cnt. Norm.)',   xy=(
                0.02, 0.94), xycoords="axes fraction", ha="left", va="top", fontsize=12)
            axs["5"].annotate('Changed Pixels',        xy=(0.02, 0.94), xycoords="axes fraction",
                              ha="left", va="top", fontsize=12, color='w',
                              bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.95))

            k_ = min(100, hist_total.size)
            mean_bottomk = np.mean(np.partition(
                hist_total.ravel(), k_ - 1)[:k_])
            mean_topk = np.mean(np.partition(hist_total.ravel(), -k_)[-k_:])
            vmin = int(mean_bottomk + 1)
            vmax = int(mean_topk + 1)

            axs["1"].imshow(hist_counts[:, y_min_hist:y_max_hist].T, origin='lower', aspect='auto',
                            vmin=vmin, vmax=vmax, cmap=colormap)
            axs["1"].set_ylabel("Pixel")
            axs["1"].grid(color="black", alpha=1)
            axs["2"].plot(np.arange(hist_counts.shape[0]), np.sum(
                hist_counts, axis=1), c='darkviolet')
            x_bin_r, y_bin_r, _ = binned_spectrum(np.arange(hist_counts.shape[0]), np.sum(
                hist_counts, axis=1), bin_size=bin_size_plot)
            axs["2"].plot(x_bin_r, y_bin_r/bin_size_plot, c='black',
                          label=f'Binned (bin size={bin_size_plot})')
            axs["2"].set_xlabel("Pixel")
            axs["2"].set_ylabel("Counts")
            axs["2"].set_xlim(0, x_max)
            axs["2"].set_yticklabels([])
            axs["2"].legend(loc="upper right", fontsize=12)

            axs["3"].imshow(hist_total[:, y_min_hist:y_max_hist].T, origin='lower', aspect='auto',
                            vmin=vmin, vmax=vmax, cmap=colormap)

            # axs["3"].axhline(y_min_hist, color='white', linestyle='-')
            # axs["3"].axhline(y_max_hist, color='white', linestyle='-')
            # axs["5"].axhline(y_min_hist, color='white', linestyle='-')
            # axs["5"].axhline(y_max_hist, color='white', linestyle='-')
            axs["4"].plot(np.arange(hist_total_2.shape[0]), np.sum(
                hist_total_2, axis=1)*bin_size_plot, c='darkviolet')
            x_bin, y_bin, _ = binned_spectrum(np.arange(hist_total_2.shape[0]),
                                              np.sum(hist_total_2, axis=1), bin_size=bin_size_plot)
            vmin_clean = 0.25*np.percentile(y_bin, 1)
            vmax_clean = 1.15*np.percentile(y_bin, 99.9)
            axs["4"].plot(x_bin, y_bin, c='black',
                          label=f'Binned (bin size={bin_size_plot})')
            axs["4"].set_ylim(vmin_clean, vmax_clean)
            axs["4"].set_xlabel("Pixel")
            axs["4"].legend(loc="upper right", fontsize=12)
            vmin_raw = 0.25*np.percentile(np.sum(hist_counts, axis=1), 1)
            vmax_raw = 1.15*np.percentile(np.sum(hist_counts, axis=1), 99)
            axs["2"].set_ylim(vmin_raw, vmax_raw)
            axs["5"].imshow(changed_mask[:, y_min_hist:y_max_hist].T, origin='lower',
                            aspect='auto', cmap=colormap)

            plt.tight_layout()
            plt.show()
        successfully_processed_datasets += 1

        if test_run:
            break

    print(
        f"Successfully processed {successfully_processed_datasets}/{total_data_sets} data set(s)."
    )
    return processed_data


def symmetrize_spectrum(
    processed_data,
    show_plots=True,
    fig_size=9,
    dark_mode=True,
    save_figure=True,
    verbose=True,
    figures_output_dir: Optional[PathLike] = None,
    calibration_parameters_path: Optional[PathLike] = None,
    analysis_parameters_path: Optional[PathLike] = None,
    scans_path: Optional[PathLike] = None,
    spec_files_path: Optional[PathLike] = None,
):
    # Fill cropped detector regions by mirror symmetrization around the elastic line.

    _build_runtime_context(
        analysis_parameters_path=analysis_parameters_path,
        scans_path=scans_path,
        spec_files_path=spec_files_path,
        verbose=verbose,
    )

    if dark_mode:
        apply_custom_plot_style()
    else:
        apply_custom_plot_style_light()

    calibration_parameters_dir = _resolve_optional_path(
        calibration_parameters_path, cwd
    )
    if verbose:
        print(
            f"Using calibration parameters lookup directory: {calibration_parameters_dir}"
        )

    figure_output_dir = _resolve_optional_path(figures_output_dir, cwd)
    if save_figure:
        figure_output_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Using figure output directory: {figure_output_dir}")

    total_data_sets = len(processed_data)
    successfully_symmetrized_datasets = 0

    for k, data_name in enumerate(processed_data.keys(), start=1):

        if verbose:
            print(
                f"Processing data set {k}/{len(processed_data)}: {data_name}")
        (m_fit, b_fit, y_max_bins, max_sigma_x, max_sigma_y,
            mCalibration, bCalibration, var_m, var_b, cov_mb) = np.load(
                calibration_parameters_dir /
            f"calibration_parameters_{data_name}.npy"
        )

        hist_total_work = deepcopy(processed_data[data_name]['histogram'])

        # ---- 1. FIT VERTICAL PROFILE ----
        x_data = np.arange(hist_total_work.shape[1])
        y_data = np.sum(hist_total_work, axis=0)

        p0 = [
            y_data.max() - y_data.min(),
            np.sum(x_data * y_data) / np.sum(y_data),
            np.sqrt(np.sum(y_data * (x_data - np.sum(x_data * y_data) /
                    np.sum(y_data))**2) / np.sum(y_data)),
            y_data.min()
        ]
        params_gauss, _ = curve_fit(gaussian, x_data, y_data, p0=p0)
        _, center_fit_original, sigma_fit, _ = params_gauss

        # ---- 2. SYMMETRIC CROP ----
        dist_to_bottom = center_fit_original
        dist_to_top = hist_total_work.shape[1] - center_fit_original
        symmetric_range = int(min(dist_to_bottom, dist_to_top))

        y_min_ = int(center_fit_original - symmetric_range) + 1
        # hist_total_work = hist_total_work[:, y_min_:int(center_fit_original + symmetric_range)]
        center_fit = center_fit_original - y_min_

        # ---- REDEFINE x_data AFTER CROPPING ----
        x_data = np.arange(hist_total_work.shape[1])

        # ---- 3. DETECT CROP BOUNDARIES (lower-right corner) ----
        hor_range = 200
        crop_index = 0
        crop_index_found = False
        for ii in range(hist_total_work.shape[0] - hor_range):
            if np.sum(hist_total_work[ii:ii+hor_range, :2]) == 0:
                crop_index = ii - 75
                crop_index_found = True
                break

        # ---- 4. DETECT CROP BOUNDARIES (upper-left corner) ----
        hor_range_top = 50
        crop_index_top = 0
        crop_index_top_found = False
        for jj in range(hist_total_work.shape[0] - hor_range_top):
            if np.sum(hist_total_work[hist_total_work.shape[0]-jj-hor_range_top:hist_total_work.shape[0]-jj, -4:]) == 0:
                crop_index_top = hist_total_work.shape[0] - jj + 75
                crop_index_top_found = True
                break
        if not crop_index_found and not crop_index_top_found:
            print("No crop boundaries detected. No symmetrization applied.")
            continue

        # ---- 5. FILL MISSING REGIONS BY MIRROR SYMMETRY ----
        cleaned_symmetrized = deepcopy(hist_total_work)

        # Lower-right corner: mirror across centerline
        for x in range(hist_total_work.shape[0]):
            y_lower = -(x - crop_index) * m_fit
            y_upper = (x - crop_index) * m_fit + 2*center_fit

            for y in range(hist_total_work.shape[1] + 1):
                if y < y_lower:
                    y_mirror = int(np.floor(2*center_fit - y))
                    if 0 <= y_mirror < hist_total_work.shape[1] and y_mirror > y_upper:
                        cleaned_symmetrized[x,
                                            y] = hist_total_work[x, y_mirror]

        # Upper-left corner: mirror across centerline
        if crop_index_top > 0:
            for x in range(hist_total_work.shape[0]):
                y_lower_top = (x - crop_index_top) * m_fit
                y_upper_top = -(x - crop_index_top) * m_fit + 2*center_fit

                for y in range(hist_total_work.shape[1]):
                    if y > y_upper_top:
                        y_mirror = int(np.ceil(2*center_fit - y))
                        if 0 <= y_mirror < hist_total_work.shape[1] and y_mirror < y_lower_top:
                            cleaned_symmetrized[x,
                                                y] = hist_total_work[x, y_mirror]

        # ---- 7. SAVE RESULTS IN PROCESSED DATA DICTIONARY ----
        processed_data[data_name]['symmetrized_histogram'] = cleaned_symmetrized
        processed_data[data_name]['y_symmetrized'] = np.sum(
            cleaned_symmetrized, axis=1)[::-1]
        successfully_symmetrized_datasets += 1

        if show_plots:

            vmin = 1
            vmax = hist_total_work.max()/5

            fig_single, ax0 = plt.subplots(
                figsize=(fig_size, fig_size/phi), nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 1]})

            # Raw spectrum
            ax0[0].imshow(hist_total_work[:, :].T, origin='lower', aspect='auto',
                          vmin=vmin, vmax=vmax, cmap="hot")
            ax0[0].set_ylabel('Pixel')
            ax0[0].set_title('Pre Symmetrization Spectrum')
            # Symmetrized spectrum with reference lines
            ax0[1].imshow(cleaned_symmetrized[:, :].T, origin='lower', aspect='auto',
                          vmin=vmin, vmax=vmax, cmap="hot")
            ax0[1].axhline(center_fit, color='black', linestyle='-',
                           linewidth=2, label='Center')
            ax0[1].axhline(center_fit, color='white', linestyle=':',
                           linewidth=2, label='Center')
            ax0[1].plot(np.arange(hist_total_work.shape[0]),
                        -(np.arange(hist_total_work.shape[0]) - crop_index)*m_fit,
                        color='lime', linewidth=2, label='Lower boundary')
            ax0[0].plot(np.arange(hist_total_work.shape[0]),
                        -(np.arange(hist_total_work.shape[0]) - crop_index)*m_fit,
                        color='lime', linewidth=2, label='Lower boundary')
            ax0[1].plot(np.arange(hist_total_work.shape[0]),
                        (np.arange(
                            hist_total_work.shape[0]) - crop_index)*m_fit + 2*center_fit,
                        color='magenta', linewidth=2, label='Mirror to center')

            if crop_index_top > 0:
                ax0[1].plot(np.arange(hist_total_work.shape[0]),
                            (np.arange(
                                hist_total_work.shape[0]) - crop_index_top)*(-m_fit) + 2*center_fit,
                            color='lime', linewidth=2, label='Upper boundary')
                ax0[0].plot(np.arange(hist_total_work.shape[0]),
                            (np.arange(
                                hist_total_work.shape[0]) - crop_index_top)*(-m_fit) + 2*center_fit,
                            color='lime', linewidth=2, label='Upper boundary')
                ax0[1].plot(np.arange(hist_total_work.shape[0]),
                            -(np.arange(hist_total_work.shape[0]) - crop_index_top)*(-m_fit),
                            color='magenta', linewidth=2, label='Mirror top')
            ax0[0].set_ylim(0, hist_total_work.shape[1])
            ax0[1].set_ylim(0, hist_total_work.shape[1])
            # ax0[1].legend(loc='upper left', fontsize=10,  framealpha=0.8, labelcolor='black',
            #                edgecolor='black', facecolor='white')
            ax0[1].set_ylabel('Pixel')
            ax0[1].set_title('Symmetrized Spectrum')

            fig_single.suptitle(
                f"Symmetrize {data_name}", fontsize=14)
            plt.tight_layout()
            plt.show()

            if save_figure:
                if dark_mode:
                    plt.savefig(
                        figure_output_dir / f"Symmetrisation_{data_name}.pdf", facecolor="#0d1117", dpi=300)
                else:
                    plt.savefig(
                        figure_output_dir / f"Symmetrisation_{data_name}_light.pdf", dpi=300)
    print(
        f"Successfully symmetrized {successfully_symmetrized_datasets}/{total_data_sets} data set(s)."
    )
    return processed_data


def export_1D_spectra(
    processed_data,
    merge=True,
    export_txt=False,
    spectra_output_dir: Optional[PathLike] = None,
):
    # Export processed 1D spectra with optional merge for numeric suffixes (e.g., _1, _2).

    output_dir = _resolve_optional_path(spectra_output_dir, cwd)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using spectra output directory: {output_dir}")

    def _save_spectrum(name, array):
        np.save(output_dir / f"spectrum_{name}.npy", array)
        if export_txt:
            np.savetxt(output_dir / f"spectrum_{name}.txt", array)
            print(
                f"Spectrum saved as spectrum_{name}.npy and spectrum_{name}.txt\n"
            )
        else:
            print(
                f"Spectrum saved as spectrum_{name}.npy\n"
            )

    per_sample_spectra = {}
    for sample_name, keys in processed_data.items():
        if 'symmetrized_histogram' in keys:
            export_histogram = keys['symmetrized_histogram']
        else:
            export_histogram = keys['histogram']

        if keys['Loss_Scale']:
            y_export = np.sum(export_histogram, axis=1)[::-1]
        else:
            y_export = np.sum(export_histogram, axis=1)

        x_export = keys['x']
        per_sample_spectra[sample_name] = np.column_stack(
            (x_export, y_export, np.sqrt(np.maximum(y_export, 0)))
        )

    if not merge:
        for sample_name, export_array in per_sample_spectra.items():
            _save_spectrum(sample_name, export_array)

        return

    grouped_names: Dict[str, List[str]] = {}
    for sample_name in processed_data.keys():
        split_name = sample_name.rsplit('_', 1)
        if len(split_name) == 2 and split_name[1].isdigit():
            base_name = split_name[0]
        else:
            base_name = sample_name

        if base_name not in grouped_names:
            grouped_names[base_name] = []
        grouped_names[base_name].append(sample_name)

    for base_name, members in grouped_names.items():
        if len(members) > 1:
            spectra_to_sum = [per_sample_spectra[name] for name in members]
            export_array = sum_spectra(spectra_to_sum)
            _save_spectrum(base_name, export_array)
            print(
                f"Merged {', '.join(members)} into {base_name}.\n"
            )
        else:
            sample_name = members[0]
            export_array = per_sample_spectra[sample_name]
            _save_spectrum(sample_name, export_array)

















































































from .helper_functions import (line, gaussian, parabola, build_lookup,
                               apply_curve_correction, get_pgm_en_graze,
                               read_scan_xy, find_scan_files, rotate_events_xy,
                               prepare_scans, _rot_to_local,_solve_y_branches,
                               _lower_upper_branch, intersection_with_mask)



def calibration_graze(
    scans_path,
    calib_scan_nums,
    scan_ranges,
    spec_dir="../SpecData",
    bottom_bound = 0,
    output_dir=None,
    show_plots=True,
    save_images=False,
    dark_mode=True,
    zoom=1,
    colormap="bone",
    figure_dpi=100,
    hist_shapes=1799,
    sigma_smooth=30,
    jump_buffer=5,
    batch_size=30,
    buffer_edges=25,
    x0=900.0,
    y0=900.0,
    x0_2=900.0,
    y0_2=975.0,
    r=850.0,
):
    
    scans_path = Path(scans_path)

    calib_scan_nums, calibration_energies, scan_energies = prepare_scans(
        calib_scan_nums,
        scan_ranges,
        spec_dir=spec_dir,
    )

    if dark_mode:
        apply_custom_plot_style()
        ending = "D"
    else:
        apply_custom_plot_style_light(grid_color="None")
        ending = "L"

    facecolor_save = "#0D1117" if dark_mode else (1, 1, 1, 0)

    if output_dir is None:
        output_dir = scans_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_bins = np.arange(hist_shapes + 1)
    y_bins = np.arange(hist_shapes + 1)

    # =========================================================
    # STEP 1: Bestimme mittleren Kippwinkel der Kalibrierlinien
    # =========================================================
    calibration_files = []
    region_limits = []
    histogram_all_raw = np.zeros((hist_shapes, hist_shapes))
    rotation_angles = []

    x_files = find_scan_files(scans_path, calib_scan_nums)
    x_files = sorted(
        x_files,
        key=lambda t: calibration_energies[calib_scan_nums.index(t[0])]
    )

    for scan_num, path_x, path_y in x_files:
        eventXY = read_scan_xy(path_x, path_y)
        calibration_files.append(eventXY)

        x = eventXY[:, 0]
        y = eventXY[:, 1]
        # Filter: only keep events with y > bottom_bound (x and y stay in sync)
        mask_bottom_bound = y > bottom_bound
        eventXY = eventXY[mask_bottom_bound]
        x = x[mask_bottom_bound]
        y = y[mask_bottom_bound]

        hist2d, _, _ = np.histogram2d(x, y, bins=(x_bins, y_bins))
        histogram_all_raw += hist2d

        counts_smooth = gaussian_filter1d(hist2d.sum(axis=0), sigma=sigma_smooth)
        deriv = np.gradient(counts_smooth)

        y_low = int(np.argmax(deriv) + jump_buffer) - jump_buffer
        y_high = int(np.argmin(deriv) - jump_buffer)

        valid_y = np.arange(y_low, y_high + 1)
        x_fit = []
        y_fit = []

        for i in range(0, len(valid_y), batch_size):
            batch = valid_y[i:i + batch_size]
            if len(batch) < batch_size:
                break

            profile_x = np.sum(hist2d[:, batch], axis=1)
            x_fit.append(float(np.argmax(profile_x)))
            y_fit.append(int(valid_y[i + batch_size // 2]))

        if len(x_fit) < 2:
            continue

        region_limits.append({
            "x_top": int(x_fit[-1]),
            "x_bottom": int(x_fit[0]),
            "bottom_jump": int(y_low),
            "top_jump": int(y_high),
        })

        popt, _ = curve_fit(line, x_fit, y_fit)
        rotation_angles.append(np.rad2deg(np.pi / 2 + np.arctan(popt[0])))

    if len(rotation_angles) == 0:
        raise ValueError("Keine gültigen Kalibrierscans für die Winkelbestimmung gefunden.")

    angle = float(np.mean(rotation_angles))

    # =========================================================
    # STEP 2: Rotierte Bilder + Parabel + Maskenregion
    # =========================================================
    histogram_all = np.zeros((hist_shapes, hist_shapes))
    calibration_files_rot = []
    region_limits = []

    for scan_num, path_x, path_y in x_files:
        eventXY = read_scan_xy(path_x, path_y)
        calibration_files_rot.append(eventXY)

        x = eventXY[:, 0]
        y = eventXY[:, 1]

        mask_bottom_bound = y > bottom_bound
        eventXY = eventXY[mask_bottom_bound]
        x = x[mask_bottom_bound]
        y = y[mask_bottom_bound]

        x, y = rotate_events_xy(x, y, -angle, xc=900.0, yc=900.0)

        hist2d, _, _ = np.histogram2d(x, y, bins=(x_bins, y_bins))
        histogram_all += hist2d

        counts_smooth = gaussian_filter1d(hist2d.sum(axis=0), sigma=sigma_smooth)
        deriv = np.gradient(counts_smooth)

        y_low = int(np.argmax(deriv) + jump_buffer) - jump_buffer
        y_high = int(np.argmin(deriv) - jump_buffer)

        valid_y = np.arange(y_low, y_high + 1)
        x_fit = []
        y_fit = []

        for i in range(0, len(valid_y), batch_size):
            batch = valid_y[i:i + batch_size]
            if len(batch) < batch_size:
                break

            profile_x = np.sum(hist2d[:, batch], axis=1)
            x_fit.append(float(np.argmax(profile_x)))
            y_fit.append(int(valid_y[i + batch_size // 2]))

        if len(x_fit) == 0:
            continue

        region_limits.append({
            "x_top": int(x_fit[-1]),
            "x_bottom": int(x_fit[0]),
            "bottom_jump": int(y_low),
            "top_jump": int(y_high),
        })

    pts_bottom = np.array(
        [[reg["x_bottom"], reg["bottom_jump"]] for reg in region_limits],
        dtype=float
    )

    if pts_bottom.shape[0] < 3:
        raise ValueError("Zu wenige Punkte für stabilen Parabelfit.")

    idx_min_x = int(np.argmin(pts_bottom[:, 0]))
    y_anchor = float(pts_bottom[idx_min_x, 1])
    dy = np.clip(y_anchor - y0_2, -r + 1e-9, r - 1e-9)
    x_anchor = float(x0_2 - np.sqrt(r**2 - dy**2))
    pts_fit = np.vstack(([x_anchor, y_anchor], pts_bottom))

    x_line = np.arange(histogram_all.shape[0], dtype=float)
    xc = float(np.mean(pts_fit[:, 0]))
    yc = float(np.mean(pts_fit[:, 1]))

    def _phi_cost(phi):
        u, v = _rot_to_local(pts_fit[:, 0], pts_fit[:, 1], phi, xc, yc)
        a, b, c = np.polyfit(u, v, 2)
        return float(np.sqrt(np.mean((v - (a * u**2 + b * u + c))**2)))

    phi = float(minimize_scalar(
        _phi_cost,
        bounds=(-np.pi / 2, np.pi / 2),
        method="bounded"
    ).x)#type: ignore

    u, v = _rot_to_local(pts_fit[:, 0], pts_fit[:, 1], phi, xc, yc)
    a_rot, b_rot, c_rot = np.polyfit(u, v, 2)

    y1, y2 = _solve_y_branches(x_line, a_rot, b_rot, c_rot, phi, xc, yc)
    y_parabola, _ = _lower_upper_branch(y1, y2)

    nx, ny = histogram_all.shape
    xg = np.arange(nx, dtype=float)
    yg = np.arange(ny, dtype=float)
    Xg, Yg = np.meshgrid(xg, yg, indexing="ij")

    circle1_inside = (Xg - x0) ** 2 + (Yg - y0) ** 2 <= r**2
    circle2_inside = (Xg - x0_2) ** 2 + (Yg - y0_2) ** 2 <= r**2

    y_parabola_grid = np.interp(xg, x_line, y_parabola, left=np.nan, right=np.nan)
    parabola_valid = np.isfinite(y_parabola_grid)
    parabola_halfspace = Yg >= y_parabola_grid[:, None]

    mask_region_raw = (
        circle1_inside
        & circle2_inside
        & parabola_halfspace
        & parabola_valid[:, None]
    )

    dist_to_boundary = distance_transform_edt(mask_region_raw)
    mask_region = mask_region_raw & (dist_to_boundary >= buffer_edges) # type:ignore
    mask_region_plot = np.ma.masked_where(~mask_region, mask_region.astype(float))

    

    # =========================================================
    # STEP 3: Pro Datei Parabelfits / ridge tracking
    # =========================================================
    parabola_vertices = []
    curve_correction = []
    failed_fits = 0
    region_limits_corrected = []

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fit_points = []
    max_gauss_width = 0

    calibration_files = calibration_files_rot

    for file_num, eventXY in enumerate(calibration_files):
        x = eventXY[:, 0]
        y = eventXY[:, 1]

        mask_bottom_bound = y > bottom_bound
        eventXY = eventXY[mask_bottom_bound]
        x = x[mask_bottom_bound]
        y = y[mask_bottom_bound]

        x, y = rotate_events_xy(x, y, -angle, xc=900.0, yc=900.0)

        xi = np.clip(x.astype(int), 0, mask_region.shape[0] - 1)
        yi = np.clip(y.astype(int), 0, mask_region.shape[1] - 1)

        keep_raw = mask_region_raw[xi, yi]
        x_raw = x[keep_raw]
        y_raw = y[keep_raw]
        histogram_raw, _, _ = np.histogram2d(x_raw, y_raw, bins=(x_bins, y_bins))

        keep = mask_region[xi, yi]
        x_in = x[keep]
        y_in = y[keep]
        histogram, _, _ = np.histogram2d(x_in, y_in, bins=(x_bins, y_bins))

        reg = region_limits[file_num]
        pts = intersection_with_mask(
            reg["x_bottom"], reg["bottom_jump"],
            reg["x_top"], reg["top_jump"],
            mask_region
        )

        if pts is not None:
            p1, p2 = pts
            region_limits_corrected.append({
                "x_bottom": p1[0],
                "bottom_jump": p1[1],
                "x_top": p2[0],
                "top_jump": p2[1],
            })

        valid_y_raw = np.arange(reg["bottom_jump"], reg["top_jump"] + 1)
        y_fit = []
        x_fit = []

        for i in range(0, len(valid_y_raw), batch_size):
            batch = valid_y_raw[i:i + batch_size]
            if len(batch) < batch_size:
                break

            arr_x = np.sum(histogram_raw[:, batch], axis=1)

            try:
                popt, _ = curve_fit(
                    gaussian,
                    np.arange(arr_x.shape[0]),
                    arr_x,
                    p0=[np.max(arr_x), int(np.argmax(arr_x)), 10.0, float(np.min(arr_x))],
                )
                max_gauss_width = max(
                    max_gauss_width,
                    2 * np.sqrt(2 * np.log(2)) * abs(popt[2])
                )
                peak_pos = float(popt[1])

                if abs(peak_pos - int(np.argmax(arr_x))) > 30:
                    failed_fits += 1
                    peak_pos = float(np.argmax(arr_x))

            except Exception:
                failed_fits += 1
                peak_pos = float(np.argmax(arr_x))

            y_fit.append(int(valid_y_raw[i + batch_size // 2]))
            x_fit.append(peak_pos)

        if len(x_fit) > 0:
            x_fit_arr = np.asarray(x_fit, dtype=float)
            y_fit_arr = np.asarray(y_fit, dtype=int)

            xi_fit = np.clip(np.rint(x_fit_arr).astype(int), 0, mask_region.shape[0] - 1)
            yi_fit = np.clip(y_fit_arr, 0, mask_region.shape[1] - 1)

            keep_fit = mask_region[xi_fit, yi_fit]
            x_fit = x_fit_arr[keep_fit].tolist()
            y_fit = y_fit_arr[keep_fit].tolist()

        if len(x_fit) >= 3:
            fit_points.append((x_fit, y_fit))

    if len(fit_points) == 0:
        raise ValueError("Keine gültigen Fitpunkte für die Curvature Correction gefunden.")

    roi = (0, hist_shapes)
    for k in range(len(fit_points)):
        roi = (max(roi[0], fit_points[k][1][0]), min(roi[1], fit_points[k][1][-1]))
    min_y, max_y = roi

    max_gauss_width = int(max_gauss_width)
    buffer_vertical = buffer_edges

    min_x = max(
        np.where(mask_region[:, max_y - buffer_vertical])[0][0] - max_gauss_width,
        np.where(mask_region[:, min_y + buffer_vertical])[0][0] - max_gauss_width,
    )
    max_x = min(
        np.where(mask_region[:, max_y - buffer_vertical])[0][-1] + max_gauss_width,
        np.where(mask_region[:, min_y + buffer_vertical])[0][-1] + max_gauss_width,
    )

    for x_fit, y_fit in fit_points:
        y_fit = np.asarray(y_fit)
        mask = (y_fit > min_y) & (y_fit < max_y)
        y_fit_use = y_fit[mask]
        x_fit_use = np.asarray(x_fit)[mask]

        if len(x_fit_use) < 3:
            continue

        popt_parabola, _ = curve_fit(parabola, y_fit_use, x_fit_use)
        a, b, c = map(float, popt_parabola)

        parabola_vertices.append(-b / (2 * a))
        center_point_vertical = (min_y + max_y) // 2
        center_parabola_x = float(parabola(center_point_vertical, a, b, c))
        curve_correction.append((center_parabola_x, a, b, c))



    # =========================================================
    # Figure 1 – jetzt wirklich wie im Notebook:
    # mit Fitpunkten, ROI-Linien und einzelnen Parabelfits
    # =========================================================
    if show_plots:
        figure_1, axes_1 = plt.subplots(
            1, 2,
            figsize=(20 * zoom, 10 * zoom),
            dpi=figure_dpi
        )

        histogram_all_plot = histogram_all.copy() + 1
        vmax = np.percentile(histogram_all_plot, 98.0)
        norm = LogNorm(vmin=1, vmax=max(vmax, 2))
        imshow_kwargs = dict(origin="lower", aspect="auto", cmap=colormap, norm=norm)

        axes_1[1].imshow(histogram_all_plot.T, **imshow_kwargs)
        axes_1[1].set_aspect("equal")

        axes_1[1].scatter(
            pts_fit[:, 0],
            pts_fit[:, 1],
            marker="P",
            s=120,
            c="darkviolet",
            label="Bottom-Punkte"
        )
        axes_1[1].plot(
            x_line, y1,
            color="darkviolet",
            lw=2.5,
            label="Gedrehte Parabel (Ast 1)"
        )
        axes_1[1].plot(
            x_line, y2,
            color="deepskyblue",
            lw=2.0,
            alpha=0.9,
            label="Gedrehte Parabel (Ast 2)"
        )

        theta = np.linspace(0, 2 * np.pi, 500)
        axes_1[1].plot(x0 + r * np.cos(theta), y0 + r * np.sin(theta), label="Circle 1")
        axes_1[1].plot(x0_2 + r * np.cos(theta), y0_2 + r * np.sin(theta), label="Circle 2")

        axes_1[1].contour(
            mask_region_plot.mask.T.astype(float),
            levels=[0.5],
            colors="lime",
            linewidths=2
        )

        axes_1[0].set_title("Calibration Lines - Raw Detector Image")
        axes_1[1].set_title("Calibration Lines - Rotated, masked Detector Image with Parabola Fits")

        histogram_all_raw_plot = histogram_all_raw.copy() + 1
        vmax_raw = np.percentile(histogram_all_raw_plot, 98.0)
        norm_raw = LogNorm(vmin=1, vmax=max(vmax_raw, 2))
        imshow_kwargs_raw = dict(origin="lower", aspect="auto", cmap=colormap, norm=norm_raw)

        axes_1[0].imshow(histogram_all_raw_plot.T, **imshow_kwargs_raw)

        axes_1[0].set_xlim(0, hist_shapes)
        axes_1[0].set_ylim(0, hist_shapes)
        axes_1[1].set_xlim(0, hist_shapes)
        axes_1[1].set_ylim(0, hist_shapes)

        axes_1[1].axvline(min_x, color="red", linestyle="--", lw=2, label="Min X Fit Points")
        axes_1[1].axvline(max_x, color="red", linestyle="--", lw=2, label="Max X Fit Points")
        axes_1[1].axhline(min_y, color="red", linestyle="--", lw=2, label="Min Y Fit Points")
        axes_1[1].axhline(max_y, color="red", linestyle="--", lw=2, label="Max Y Fit Points")

        for file_num, (x_fit, y_fit) in enumerate(fit_points):
            y_fit_arr = np.array(y_fit)
            mask_fit_roi = (y_fit_arr > min_y) & (y_fit_arr < max_y)
            y_fit_roi = y_fit_arr[mask_fit_roi]
            x_fit_roi = np.array(x_fit)[mask_fit_roi]

            if len(x_fit_roi) < 3:
                continue

            popt_parabola, _ = curve_fit(parabola, np.asarray(y_fit_roi), x_fit_roi)
            a, b, c = map(float, popt_parabola)

            axes_1[1].plot(
                x_fit_roi,
                y_fit_roi,
                "o",
                markeredgecolor="black",
                color="white"
            )
            axes_1[1].plot(
                x_fit_roi,
                y_fit_roi,
                "o",
                markersize=4,
                color=colors[file_num % len(colors)]
            )

            axes_1[1].plot(
                parabola(np.arange(hist_shapes), a, b, c),
                np.arange(hist_shapes),
                "-",
                color=colors[file_num % len(colors)]
            )

        plt.suptitle("Diagnostics 1 - Masking, valid regions and curvature correction")
        plt.tight_layout()
        plt.show()
        if save_images:
            plt.savefig(
                output_dir / f"Calibration_Graze_Diagnostics_1_{ending}.pdf",
                facecolor=facecolor_save,
                dpi=300
            )







    

    if len(curve_correction) < 2:
        raise ValueError("Zu wenige gültige Kurven für Lookup-Interpolation.")

    curve_correction = np.array(copy.deepcopy(curve_correction))
    x_centers = curve_correction[:, 0]
    a_vals = curve_correction[:, 1]
    b_vals = curve_correction[:, 2]
    c_vals = curve_correction[:, 3]

    a_interp = interp1d(x_centers, a_vals, kind="linear", bounds_error=False, fill_value="extrapolate") #type: ignore
    b_interp = interp1d(x_centers, b_vals, kind="linear", bounds_error=False, fill_value="extrapolate") #type: ignore
    c_interp = interp1d(x_centers, c_vals, kind="linear", bounds_error=False, fill_value="extrapolate") #type: ignore

    x_center_grid = np.arange(hist_shapes + 1)
    y_grid = np.arange(hist_shapes + 1)

    a_grid = a_interp(x_center_grid)
    b_grid = b_interp(x_center_grid)
    c_grid = c_interp(x_center_grid)

    X_field = (
        a_grid[:, None] * y_grid[None, :] ** 2
        + b_grid[:, None] * y_grid[None, :]
        + c_grid[:, None]
    )

    lookup = build_lookup(X_field, hist_shapes)

    # =========================================================
    # STEP 4: Apply curvature correction to calibration scans
    # =========================================================
    hist_corr_list = []

    for file_num, eventXY in enumerate(calibration_files):
        x = eventXY[:, 0]
        y = eventXY[:, 1]
        x, y = rotate_events_xy(x, y, -angle, xc=900.0, yc=900.0)

        xi = np.clip(x.astype(int), 0, mask_region.shape[0] - 1)
        yi = np.clip(y.astype(int), 0, mask_region.shape[1] - 1)

        keep = mask_region[xi, yi]
        x_in = x[keep]
        y_in = y[keep]

        hist2d, _, _ = np.histogram2d(
            x,
            y,
            bins=(np.arange(hist_shapes + 1), np.arange(hist_shapes + 1)),
        )

        hist_corr = apply_curve_correction(hist2d, lookup, x_center_grid)
        hist_corr_list.append(hist_corr)

    histogram_sum = np.sum(hist_corr_list, axis=0)

    # =========================================================
    # Figure 2 – exakt wie dein Notebook
    # =========================================================
    amplitudes = []
    gauss_centers = []
    sigmas = []
    offsets = []

    x_vals = np.arange(hist_corr_list[0].shape[0]) if hist_corr_list else np.arange(hist_shapes)

    histogram_height = (min(max_y + 100, hist_shapes) - max(min_y - 100, 0)) / hist_shapes

    if show_plots:
        figure_2, ax = plt.subplots(
            figsize=(10 * zoom, (10 * histogram_height + 5) * zoom),
            ncols=1,
            nrows=3,
            gridspec_kw={"height_ratios": [10 * histogram_height, 2.5, 2.5]},
            dpi=figure_dpi,
        )

        histogram_sum_plot = histogram_sum.copy() + 1
        vmax = np.percentile(histogram_sum_plot, 98.0)
        norm = LogNorm(vmin=1, vmax=max(vmax, 2))

        ax[0].imshow(
            histogram_sum_plot[:, max(0, min_y - 100):min(max_y + 100, hist_shapes)].T,
            origin="lower",
            aspect="auto",
            norm=norm,
            cmap=colormap,
            extent=[0, histogram_sum.shape[0], max(0, min_y - 100), min(max_y + 100, hist_shapes)]
        )

        ax[0].axhline(min_y, color="red", ls="--", lw=2.0)
        ax[0].axhline(max_y, color="red", ls="--", lw=2.0)
        ax[0].axvline(min_x, color="red", ls="--", lw=2.0)
        ax[0].axvline(max_x, color="red", ls="--", lw=2.0)

        for count, xi_c in enumerate(x_centers):
            color = colors[count % len(colors)]
            ax[0].axvline(xi_c, color=color, linewidth=2, alpha=1)

        ax[0].set_title("Calibration Lines - After Curve Correction")

    for hist in hist_corr_list:
        profile = np.sum(hist[:, min_y:max_y], axis=1)

        A0 = max(float(np.max(profile) - np.min(profile)), 1e-12)
        mu0 = float(np.argmax(profile))
        sigma0 = float(np.sqrt(np.sum((x_vals - mu0) ** 2 * profile) / np.sum(profile)))
        y00 = float(np.min(profile))
        p0 = [A0, mu0, max(sigma0, 1.0), y00]

        bounds = (
            [1e-12, 0, 1e-12, 0],
            [max(1.5 * A0, 1.0), 1800, 600, max(A0 / 10.0, 1.0)]
        )

        try:
            popt, _ = curve_fit(gaussian, x_vals, profile, p0=p0, bounds=bounds, maxfev=10000)
        except Exception:
            popt = p0

        A, mu, sigma, y0_fit = popt
        A_pos = max(float(A), 1e-12)
        sigma_pos = max(float(sigma), 1e-12)

        amplitudes.append(A_pos)
        gauss_centers.append(mu)
        sigmas.append(sigma_pos)
        offsets.append(y0_fit)

        if show_plots:
            l_lim = int(max(0, (mu - 4 * sigma_pos)))
            r_lim = int(min(1800, mu + 4 * sigma_pos))

            ax[1].plot( # type:ignore
                np.arange(hist.shape[0])[l_lim:r_lim],
                np.sum(hist[:, min_y:max_y], axis=1)[l_lim:r_lim],
                ".",
                color="black",
            )
            ax[1].plot( # type:ignore
                x_vals[l_lim:r_lim],
                gaussian(x_vals, *popt)[l_lim:r_lim],
                alpha=1,
                lw=2
            )

    amplitudes = np.array(amplitudes)
    gauss_centers = np.array(gauss_centers)
    sigmas = np.array(sigmas)
    offsets = np.array(offsets)
    FWHMs = 2 * np.sqrt(2 * np.log(2)) * sigmas

    if show_plots:
        ax[1].set_title("Integrated Lines and Fits") # type:ignore

    if len(calibration_energies) == len(gauss_centers):
        poly_deg = 3
        poly_coeffs = np.polyfit(gauss_centers, calibration_energies, poly_deg)
        poly_fn_calibration = np.poly1d(poly_coeffs)

        if show_plots:
            for k, (center, energy) in enumerate(zip(gauss_centers, calibration_energies)):
                ax[2].scatter(center, energy, marker="x", s=70, color=colors[k % len(colors)]) # type:ignore

            ax[2].plot(np.arange(0, 1800), poly_fn_calibration(np.arange(1800)), color="gray") # type:ignore
            ax[2].set_xlabel("Pixel") # type:ignore
            ax[2].set_ylabel("Energy (eV)") # type:ignore
            ax[2].set_title("Energy(Pixel) - 3rd Order Poly Fit") # type:ignore
    else:
        poly_fn_calibration = np.poly1d([0, 1, 0, 0])

        if show_plots:
            ax[2].set_title("Energy calibration requires 'calibration_energies'.") # type:ignore

    if show_plots:
        ax[0].set_xlim(max(0, min_x - 100), min(max_x + 100, hist_shapes)) # type:ignore
        ax[1].set_xlim(max(0, min_x - 100), min(max_x + 100, hist_shapes)) # type:ignore
        ax[2].set_xlim(max(0, min_x - 100), min(max_x + 100, hist_shapes)) # type:ignore

        plt.tight_layout()
        plt.suptitle("Diagnostics 2 - Curve Correction Results and Energy Calibration", y=1.02)
        if save_images:
            figure_2.savefig(output_dir / f"Calibration_Graze_Diagnostics_2_{ending}.pdf", dpi=figure_dpi, facecolor=facecolor_save) #type: ignore
        plt.show()

    calibration = {
        "scans_path": scans_path,
        "spec_dir": spec_dir,
        "calib_scan_nums": calib_scan_nums,
        "scan_ranges": scan_ranges,
        "calibration_energies": calibration_energies,
        "scan_energies": scan_energies,
        "angle": angle,
        "lookup": lookup,
        "x_center_grid": x_center_grid,
        "poly_fn_calibration": poly_fn_calibration,
        "poly_coeffs": poly_fn_calibration.coefficients,
        "min_x": int(min_x),
        "max_x": int(max_x),
        "min_y": int(min_y),
        "max_y": int(max_y),
        "mask_region": mask_region,
        "mask_region_raw": mask_region_raw,
        "fit_points": fit_points,
        "curve_correction": curve_correction,
        "gauss_centers": gauss_centers,
        "FWHMs": FWHMs,
        "hist_corr_list": hist_corr_list,
        "histogram_all_raw": histogram_all_raw,
        "histogram_all_rot": histogram_all,
        "ending": ending,
        "facecolor_save": facecolor_save,
        "dark_mode": dark_mode,
        "colormap": colormap,
        "figure_dpi": figure_dpi,
        "hist_shapes": hist_shapes,
    }

    return calibration


def process_RIXS_graze(
    calibration,
    scan_ranges=None,
    show_plots=True,
    bottom_bound=0,
    dark_mode=None,
    save_images=False,
    save_txt=True,
    output_dir=None,
    zoom=1,
    colormap="bone",
    figure_dpi=None,
    correct_energy=True,
    auto_roi=True,
    anti_raman=False,
    artifacts=None,
    artifacts_region=None,
    inset=False,
    bin_size=1,
):
    
    scans_path = Path(calibration["scans_path"])
    spec_dir = calibration["spec_dir"]

    angle = calibration["angle"]
    lookup = calibration["lookup"]
    x_center_grid = calibration["x_center_grid"]
    poly_fn_calibration = calibration["poly_fn_calibration"]

    hist_shapes = calibration["hist_shapes"]
    min_x = calibration["min_x"]
    max_x = calibration["max_x"]

    if scan_ranges is None:
        scan_ranges = calibration["scan_ranges"]

    if dark_mode is None:
        dark_mode = calibration.get("dark_mode", True)

    if figure_dpi is None:
        figure_dpi = calibration.get("figure_dpi", 100)

    if dark_mode:
        apply_custom_plot_style()
        ending = "D"
    else:
        apply_custom_plot_style_light(grid_color="None")
        ending = "L"

    facecolor_save = "#0D1117" if dark_mode else (1, 1, 1, 0)

    if output_dir is None:
        output_dir = scans_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scan_energies = [
        np.mean(list(get_pgm_en_graze(r, spec_dir=spec_dir).values()))
        for r in scan_ranges
    ]

    if auto_roi:
        min_y = calibration["min_y"]
        max_y = calibration["max_y"]
    else:
        min_y, max_y = 650, 1450

    if artifacts_region is None:
        if artifacts is not None:
            artifacts_region = [[(900, max_y)]] * len(scan_ranges)
        else:
            artifacts_region = [None] * len(scan_ranges)

    if len(artifacts_region) != len(scan_ranges):
        raise ValueError("artifacts_region muss dieselbe Laenge wie scan_ranges haben.")

    dictionary_results = {}

    for num, measurement in enumerate(scan_ranges):
        scan_energy = scan_energies[num]

        scan_nums = np.arange(measurement[0], measurement[1] + 1)
        scan_files = find_scan_files(scans_path, scan_nums)

        hist_corr_list = []

        for _, path_x, path_y in scan_files:
            eventXY = read_scan_xy(path_x, path_y)
            x, y = eventXY[:, 0], eventXY[:, 1]
            mask_bottom_bound = y > bottom_bound
            eventXY = eventXY[mask_bottom_bound]
            x = x[mask_bottom_bound]
            y = y[mask_bottom_bound]
            x, y = rotate_events_xy(x, y, -angle, xc=900.0, yc=900.0)

            hist2d, _, _ = np.histogram2d(
                x,
                y,
                bins=(np.arange(hist_shapes + 1), np.arange(hist_shapes + 1)),
            )

            hist_corr = apply_curve_correction(hist2d, lookup, x_center_grid)
            hist_corr_list.append(hist_corr)

        if len(hist_corr_list) == 0:
            print(f"Keine Dateien gefunden fuer Scanbereich {measurement}.")
            continue

        histogram_sum = np.sum(hist_corr_list, axis=0)
        histogram_sum_raw = histogram_sum.copy()

        if artifacts_region[num] is not None:
            for region in artifacts_region[num]: # type: ignore
                histogram_sum[:, region[0]:region[1]] = 0

        histogram_sum += 1
        histogram_sum_raw += 1

        vmax = np.percentile(histogram_sum_raw, 98.0)
        norm = LogNorm(vmin=1, vmax=max(vmax, 2))

        fit_success = True
        shift = 0.0
        popt = None
        fit_y = None
        label_fit = "Elastic Line Gauss Fit"

        x_proj = poly_fn_calibration(np.arange(histogram_sum.shape[0]))
        y_proj = histogram_sum[:, min_y:max_y].sum(axis=1)

        if correct_energy:
            min_x_val = int(np.argmax(histogram_sum.sum(axis=1)))

            if anti_raman and max_x - min_x_val > 15:
                min_e_val = poly_fn_calibration(min_x_val)
                mask_fit = x_proj >= min_e_val
                x_proj_ar = x_proj[mask_fit]
                y_proj_ar = y_proj[mask_fit]
                label_fit = "Gauss Fit (Anti-Raman Flank)"
            else:
                x_proj_ar = x_proj
                y_proj_ar = y_proj
                label_fit = "Elastic Line Gauss Fit"

            A0 = y_proj_ar.max() - y_proj_ar.min()
            mu0 = x_proj_ar[np.argmax(y_proj_ar)]
            sigma0 = (x_proj_ar[-1] - x_proj_ar[0]) / 20
            y00 = y_proj_ar.min()
            p0 = [A0, mu0, sigma0, y00]

            try:
                popt, pcov = curve_fit(gaussian, x_proj_ar, y_proj_ar, p0=p0)
                fit_y = gaussian(x_proj, *popt)

                shift = float(scan_energy - popt[1])

            except Exception:
                fit_success = False
                shift = 0.0
                popt = None
                fit_y = None

        energy_axis = poly_fn_calibration(np.arange(min_x, max_x)) + shift
        counts_axis = histogram_sum[min_x:max_x, min_y:max_y].sum(axis=1)

        binned_energy, binned_counts, binned_sigma = binned_spectrum(
            energy_axis,
            counts_axis,
            bin_size,
        )

        if show_plots:
            fig_rixs, ax = plt.subplots(
                2,
                1,
                figsize=(8 * zoom, 8 * zoom),
                dpi=figure_dpi,
            )
            fig_rixs.suptitle(
                f"Measurement(s) {measurement[0]}-{measurement[-1]} at nominal energy {round(scan_energy, 2)} eV"
            )

            ax[0].imshow(
                histogram_sum_raw[min_x:max_x, min_y:max_y].T,
                origin="lower",
                aspect="auto",
                cmap=colormap,
                norm=norm,
                extent=[min_x, max_x, min_y, max_y]
            )

            ax[0].axhline(min_y, color="red", ls="--", lw=2)
            ax[0].axhline(max_y, color="red", ls="--", lw=2)
            ax[0].axvline(min_x, color="red", ls="--", lw=2)
            ax[0].axvline(max_x, color="red", ls="--", lw=2)
            
            ax[0].set_title("Curve-corrected summed detector image")

            ax[1].plot(
                energy_axis,
                counts_axis,
                lw=1.5,
                label="Unbinned spectrum", zorder = 1
            )

            if bin_size > 1:
                ax[1].plot(
                    binned_energy,
                    binned_counts / bin_size,
                    ".-",
                    label=f"Binned spectrum (bin={bin_size})",
                )

            if correct_energy and fit_success and popt is not None and fit_y is not None:
                std_deviations = 20
                mask_x = (
                    (x_proj >= popt[1] - std_deviations * abs(popt[2]))
                    & (x_proj <= popt[1] + std_deviations * abs(popt[2]))
                )

                ax[1].plot(
                    (x_proj + shift)[mask_x],
                    fit_y[mask_x],
                    lw=2,
                    label=label_fit, zorder = 0
                )

                ax[1].axvline(
                    scan_energy,
                    color="red",
                    ls="--",
                    lw=1.5,
                    label=f"Nominal energy = {scan_energy:.2f} eV",
                )

            ax[1].set_xlabel("Energy (eV)")
            ax[1].set_ylabel("Counts")
            ax[1].set_title(f"Final RIXS spectrum (shift = {shift:.3f} eV)")
            ax[1].legend()
            ax[1].set_xlim(energy_axis[0], energy_axis[-1])

            plt.tight_layout()
            plt.show()

            if save_images:
                out_pdf = output_dir / f"RIXS_{measurement[0]}_{measurement[1]}_{ending}.pdf"
                fig_rixs.savefig(
                    out_pdf,
                    facecolor=facecolor_save,
                    dpi=300,
                    bbox_inches="tight",
                )

        if save_txt:
            out_txt = output_dir / f"RIXS_{measurement[0]}_{measurement[1]}.txt"
            np.savetxt(
                out_txt,
                np.column_stack((binned_energy, binned_counts / bin_size, binned_sigma / bin_size)),
                header="Energy_eV\tCounts\tSigma",
            )

        dictionary_results[num] = {
            "measurement": measurement,
            "scan_files": scan_files,
            "scan_energy": scan_energy,
            "hist_corr_list": hist_corr_list,
            "histogram_sum": histogram_sum,
            "histogram_sum_raw": histogram_sum_raw,
            "energy_axis": energy_axis,
            "counts_axis": counts_axis,
            "binned_energy": binned_energy,
            "binned_counts": binned_counts / bin_size,
            "binned_sigma": binned_sigma / bin_size,
            "shift": shift,
            "fit_success": fit_success,
            "fit_params": popt,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
        }

    return dictionary_results