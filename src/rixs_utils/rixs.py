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
from copy import deepcopy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from matplotlib.axes import Axes
import math
from dataclasses import dataclass
phi = (1 + math.sqrt(5)) / 2

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
