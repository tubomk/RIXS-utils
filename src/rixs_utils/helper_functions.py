import os
import re
from collections import defaultdict
from pathlib import Path
from scipy.special import wofz
import numpy as np
from numba import njit, prange, set_num_threads

num_cpus = os.cpu_count() or 1
set_num_threads(max(1, num_cpus - 2))  # type: ignore


def parse_analysis_parameters(filepath):
    data_raw = defaultdict(dict)
    number_measurements = None
    single_entry_keys = {}

    with open(filepath, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

            match = re.match(r"(\w+)\s*=\s*(.+)", line)
            if not match:
                continue

            key, value = match.groups()

            if key == "numberMeasurements":
                number_measurements = int(value)
                continue

            match_indexed = re.match(r"(.+?)_(\d+)$", key)
            if match_indexed:
                base_key, index = match_indexed.groups()
                target = data_raw[int(index)]
            else:
                base_key = key
                target = single_entry_keys

            try:
                if value.startswith("[") or value.startswith("("):
                    value = eval(value)
                else:
                    value = value.strip('"')
            except Exception as exc:
                print(f"Error in line: {line}\n{exc}")
                continue

            target[base_key] = value

    if number_measurements is None:
        if single_entry_keys:
            number_measurements = 1
            data_raw[1] = single_entry_keys
        else:
            number_measurements = len(data_raw)

    data = []
    required_keys = ["calibrationScanNums", "scanNums"]
    ensure_list_of_tuples = ["scanNums", "calibrationScanNums"]

    for idx in sorted(data_raw.keys()):
        entry = data_raw[idx]

        for key in required_keys:
            if key not in entry:
                raise ValueError(
                    f"Missing required field '{key}' in measurement {idx}.")

        for key in ensure_list_of_tuples:
            value = entry.get(key)
            if isinstance(value, tuple) and all(isinstance(x, int) for x in value):
                if len(value) != 2:
                    raise ValueError(
                        f"Tuple for '{key}' in measurement {idx} must contain exactly 2 integers."
                    )
                value = [value]
            elif isinstance(value, list):
                for element in value:
                    if not (
                        isinstance(element, tuple)
                        and len(element) == 2
                        and all(isinstance(x, int) for x in element)
                    ):
                        raise ValueError(
                            f"Each entry in '{key}' of measurement {idx} must be a tuple of two integers."
                        )
            else:
                raise ValueError(
                    f"Invalid format for '{key}' in measurement {idx}. Must be a tuple or list of 2-tuples."
                )
            entry[key] = value

        entry.setdefault("calibrationEnergies", None)
        entry.setdefault("incidentEnergy", None)
        entry.setdefault("description", f"Data_{idx}: No description provided")
        entry.setdefault("shortDescription", f"Data_{idx}")
        data.append(entry)

    return data


def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + offset


def line(x, m, b):
    return m * x + b


def _build_cross_platform_font_rcparams():
    import matplotlib.font_manager as font_manager

    preferred_serif_fonts = [
        "Times New Roman",
        "Liberation Serif",
        "Nimbus Roman",
        "Nimbus Roman No9 L",
        "STIX Two Text",
        "STIXGeneral",
        "DejaVu Serif",
    ]

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    serif_fallbacks = [
        font_name for font_name in preferred_serif_fonts if font_name in available_fonts]

    if not serif_fallbacks:
        serif_fallbacks = ["DejaVu Serif"]

    font_rcparams = {
        "font.family": "serif",
        "font.serif": serif_fallbacks,
    }

    if "Times New Roman" in serif_fallbacks:
        font_rcparams.update(
            {
                "mathtext.fontset": "custom",
                "mathtext.rm": "Times New Roman",
                "mathtext.it": "Times New Roman:italic",
                "mathtext.bf": "Times New Roman:bold",
            }
        )
    else:
        font_rcparams.update({"mathtext.fontset": "stix"})

    return font_rcparams


def apply_custom_plot_style():
    import matplotlib.pyplot as plt

    rcparams = {
        "savefig.transparent": False,
        "axes.facecolor": (1, 1, 1, 0.15),
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "text.color": "white",
        "grid.color": "white",
        "grid.alpha": 0.3,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "axes.edgecolor": "white",
        "legend.framealpha": 0.05,
        "legend.edgecolor": "white",
        "legend.fontsize": 14,
        "legend.markerscale": 1.8,
        "legend.handlelength": 2,
        "legend.handleheight": 2,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 18,
    }
    rcparams.update(_build_cross_platform_font_rcparams())

    try:
        get_ipython()  # type: ignore
        rcparams["figure.facecolor"] = "none"
        plt.rcParams.update(rcparams)

        from IPython.display import HTML, display  # type: ignore

        css = """
        <style>
        .cell-output-ipywidget-background { background-color: transparent !important; }
        :root {
            --jp-widgets-color: var(--vscode-editor-foreground);
            --jp-widgets-font-size: var(--vscode-editor-font-size);
        }
        </style>
        """
        display(HTML(css))

    except NameError:
        rcparams["figure.facecolor"] = "#0d1117"
        plt.rcParams.update(rcparams)


def apply_custom_plot_style_light(grid_color="black"):
    import matplotlib.pyplot as plt

    rcparams = {
        "savefig.transparent": False,
        "axes.facecolor": (0, 0, 0, 0.03),
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "text.color": "black",
        "grid.color": f"{grid_color}",
        "grid.alpha": 0.18,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "axes.edgecolor": "black",
        "legend.framealpha": 0.12,
        "legend.edgecolor": "black",
        "legend.fontsize": 14,
        "legend.markerscale": 1.8,
        "legend.handlelength": 2,
        "legend.handleheight": 2,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 18,
    }
    rcparams.update(_build_cross_platform_font_rcparams())

    try:
        get_ipython()  # type: ignore
        rcparams["figure.facecolor"] = "none"
        plt.rcParams.update(rcparams)

        from IPython.display import HTML, display  # type: ignore

        css = """
        <style>
        .cell-output-ipywidget-background { background-color: transparent !important; }
        :root {
            --jp-widgets-color: var(--vscode-editor-foreground, black);
            --jp-widgets-font-size: var(--vscode-editor-font-size, 13px);
        }
        </style>
        """
        display(HTML(css))

    except NameError:
        rcparams["figure.facecolor"] = "white"
        plt.rcParams.update(rcparams)


def make_mask(shape, line_params, max_sigma, width_factor=3.5):
    ny, nx = shape
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]

    slope, intercept = line_params
    a = -slope
    b = 1
    c = -intercept

    dist = np.abs(a * x_grid + b * y_grid + c) / np.sqrt(a**2 + b**2)
    limit = width_factor * max_sigma
    return dist <= limit


def shear_and_crop_along_line(histogram, m_fit, b_fit, max_sigma_y, width_factor=3.8):
    ny, nx = histogram.T.shape
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    dist = np.abs(-m_fit * x_grid + y_grid - b_fit) / \
        np.sqrt((-m_fit) ** 2 + 1)
    mask = dist <= (width_factor * max_sigma_y)
    histogram = histogram * mask.T

    ny, nx = histogram.shape
    y0 = int(round(b_fit))
    shifts = np.empty(ny, dtype=int)
    for row_index in range(ny):
        shifts[row_index] = int(round(m_fit * row_index + b_fit - y0))

    min_shift = int(shifts.min())
    max_shift = int(shifts.max())
    new_nx = nx + (max_shift - min_shift)

    sheared = np.zeros((ny, new_nx), dtype=histogram.dtype)
    for row_index in range(ny):
        start = max_shift - shifts[row_index]
        sheared[row_index, start:start + nx] = histogram[row_index]

    return sheared


def binned_spectrum(x, y, bin_size):
    num_bins = len(x) // bin_size
    x = x[: num_bins * bin_size]
    y = y[: num_bins * bin_size]

    edges = np.arange(0, num_bins * bin_size, bin_size)
    binned_counts = np.add.reduceat(y, edges)
    binned_x = x.reshape(num_bins, bin_size).mean(axis=1)
    binned_sigma = np.sqrt(binned_counts)

    return binned_x, binned_counts, binned_sigma


def get_pgm_en(*args, spec_dir=None):
    scan_nums = []
    for arg in args:
        if isinstance(arg, (list, tuple, np.ndarray)):
            scan_nums.extend(list(arg))
        else:
            scan_nums.append(arg)

    spec_root = Path(spec_dir) if spec_dir is not None else Path.cwd()
    pgm_en_dict = {}

    for file_path in spec_root.glob("*.spec"):
        with file_path.open(encoding="utf-8") as file_obj:
            content = file_obj.read()

        scans = content.split("#S ")
        all_scan_nums = [int(scans[idx].split("\n")[0].split()[0])
                         for idx in range(1, len(scans))]

        headers = None
        for line in content.split("\n"):
            if line.startswith("#O0"):
                headers = line.split()[1:]
                break

        if headers is None or "pgm_en" not in headers:
            continue

        idx_pgm_en = headers.index("pgm_en")

        for scan_num in scan_nums:
            if scan_num not in all_scan_nums:
                continue
            scan = scans[all_scan_nums.index(scan_num) + 1]
            for line in scan.split("\n"):
                if line.startswith("#P0"):
                    data = line.split()[1:]
                    pgm_en_dict[scan_num] = float(data[idx_pgm_en])
                    break

    return [pgm_en_dict[num] if num in pgm_en_dict else None for num in scan_nums]


@njit
def reflect_index(i, n):
    if n <= 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i - 1
        else:
            i = 2 * n - i - 1
    return i


@njit(parallel=True)
def median_filter_numpy(arr, size):
    if size % 2 == 0:
        raise ValueError("size must be odd")
    pad = size // 2
    h, w = arr.shape
    out = np.empty_like(arr)

    for i in prange(h):
        win = np.empty(size * size - 1, dtype=arr.dtype)
        for j in range(w):
            idx = 0
            for di in range(-pad, pad + 1):
                ii = reflect_index(i + di, h)
                for dj in range(-pad, pad + 1):
                    jj = reflect_index(j + dj, w)
                    if ii == i and jj == j:
                        continue
                    win[idx] = arr[ii, jj]
                    idx += 1
            sorted_win = np.sort(win)
            out[i, j] = sorted_win[sorted_win.size // 2]
    return out


def replace_lines_with_neighbor_mean(hist, line_indices, window_size, dead_or_hyperactive, verbose=False):
    half_size = window_size // 2
    if line_indices.size == 0:
        return hist

    if verbose:
        print(
            f"Identified {len(line_indices)} potential {dead_or_hyperactive} line(s)")

    dead_set = set(int(index) for index in line_indices)
    for x_pos in line_indices:
        if half_size < x_pos < len(hist) - half_size - 1:
            substitute = np.zeros_like(hist[x_pos])
            valid_neighbors = 0
            for neighbor in range(x_pos - 1, x_pos + 2):
                if neighbor != x_pos and neighbor not in dead_set:
                    substitute += hist[neighbor]
                    valid_neighbors += 1
            if valid_neighbors > 0:
                hist[x_pos] = np.round(substitute / valid_neighbors)
    return hist


def print_analysis_parameters_template():
    print(
        "numberMeasurements = N\n\n"
        "calibrationScanNums_1 = [(start1, end1), (start2, end2)]\n"
        "calibrationEnergies_1 = [E1, E2, E3, ...]\n"
        "scanNums_1 = [(start, end)]\n"
        "incidentEnergy_1 = Ein\n"
        "description_1 = \"...\"\n"
        "shortDescription_1 = \"...\"\n\n"
        "calibrationScanNums_2 = [(start1, end1), (start2, end2)]\n"
        "calibrationEnergies_2 = [E1, E2, E3, ...]\n"
        "scanNums_2 = [(start, end)]\n"
        "incidentEnergy_2 = Ein\n"
        "description_2 = \"...\"\n"
        "shortDescription_2 = \"...\"\n"
        "\n\n\n"
        "NOTE: Actual calibration energies and incoming energies are optional.\n"
        "They are inferred from metadata if available. Place the corresponding .spec "
        "file in Spec_Files (or provide the directory explicitly)."
    )


def sum_spectra(*spectra, dx=None, overlap_only=True):
    # Sum multiple spectra on a common x-grid while conserving counts.
    # Expected input shape per spectrum: (N, 3) with columns [x, y, yerr].

    # Allow a single list/tuple input.
    if len(spectra) == 1 and isinstance(spectra[0], (list, tuple)):
        spectra = tuple(spectra[0])

    if len(spectra) < 2:
        raise ValueError("Provide at least two spectra of shape (N, 3).")

    # Normalize inputs and collect spans/spacings.
    xs, ys, errs = [], [], []
    xmins, xmaxs, all_dx = [], [], []

    for idx, spectrum in enumerate(spectra):
        spectrum = np.asarray(spectrum, float)
        if spectrum.ndim != 2 or spectrum.shape[1] != 3:
            raise ValueError(f"Spectrum #{idx} must have shape (N, 3).")

        x, y, yerr = spectrum[:, 0], spectrum[:, 1], spectrum[:, 2]

        if x.size < 2:
            raise ValueError(f"Spectrum #{idx} needs at least 2 points.")

        # Ensure ascending x-axis.
        if x[0] > x[-1]:
            x = x[::-1]
            y = y[::-1]
            yerr = yerr[::-1]

        xs.append(x)
        ys.append(y)
        errs.append(yerr)

        xmins.append(x[0])
        xmaxs.append(x[-1])
        dxi = np.diff(x)
        all_dx.append(dxi[(dxi > 0) & np.isfinite(dxi)])

    # Choose common target grid.
    if overlap_only:
        xmin = max(xmins)
        xmax = min(xmaxs)
        if not (xmax > xmin):
            raise ValueError(
                "No mutual overlap between spectra; cannot build a common grid.")
    else:
        xmin = min(xmins)
        xmax = max(xmaxs)

    all_dx = np.concatenate([d for d in all_dx if d.size])
    if dx is None:
        if all_dx.size == 0:
            raise ValueError(
                "Cannot infer dx from inputs; please provide dx explicitly.")
        dx = float(np.median(all_dx))

    # Validate grid spacing.
    if dx <= 0 or not np.isfinite(dx):
        raise ValueError(f"Invalid dx={dx}. Must be positive and finite.")

    n = int(np.floor((xmax - xmin) / dx)) + 1
    xgrid = np.linspace(xmin, xmin + (n - 1) * dx, n)

    # Derive target bin edges from centers.
    if xgrid.size < 2:
        raise ValueError("Common grid would have fewer than 2 bins.")
    xm = (xgrid[:-1] + xgrid[1:]) / 2.0
    dx0 = xgrid[1] - xgrid[0]
    dx1 = xgrid[-1] - xgrid[-2]
    edges = np.concatenate(
        ([xgrid[0] - dx0 / 2.0], xm, [xgrid[-1] + dx1 / 2.0]))

    # Accumulate rebinned counts and variances.
    y_sum = np.zeros_like(xgrid)
    var_sum = np.zeros_like(xgrid)

    for x, y, yerr in zip(xs, ys, errs):
        # Derive source bin edges for this spectrum.
        xm_old = (x[:-1] + x[1:]) / 2.0
        dx0_old = x[1] - x[0]
        dx1_old = x[-1] - x[-2]
        edges_old = np.concatenate(
            ([x[0] - dx0_old / 2.0], xm_old, [x[-1] + dx1_old / 2.0]))
        w_old = np.diff(edges_old)

        # Use yerr^2 where valid, otherwise fall back to Poisson variance.
        var = y.copy()
        if yerr is not None:
            var_from_yerr = yerr**2
            mask_ok = np.isfinite(var_from_yerr) & (var_from_yerr > 0)
            var[mask_ok] = var_from_yerr[mask_ok]
        var = np.clip(var, 0.0, np.inf)

        # Rebin with overlap-based two-pointer sweep.
        out_y = np.zeros_like(xgrid)
        out_var = np.zeros_like(xgrid)

        i = j = 0
        last_old = len(x) - 1
        last_new = len(xgrid) - 1

        while i < last_old and j < last_new:
            left = max(edges_old[i],   edges[j])
            right = min(edges_old[i+1], edges[j+1])

            if right > left:
                frac = (right - left) / w_old[i]
                out_y[j] += y[i] * frac
                out_var[j] += var[i] * (frac**2)

            # Advance whichever bin edge ends first.
            if edges_old[i+1] <= edges[j+1]:
                i += 1
            else:
                j += 1

        y_sum += out_y
        var_sum += out_var

    yerr_sum = np.sqrt(var_sum)
    return np.stack((xgrid, y_sum, yerr_sum), axis=1)


def voigt_norm(x, x0, sigma, gamma):
    # robuste Parameter (verhindert divide-by-zero / NaNs)
    sigma = float(sigma)
    gamma = float(gamma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1e-9
    if not np.isfinite(gamma) or gamma < 0.0:
        gamma = 0.0
    den = sigma * np.sqrt(2.0)
    z = ((x - x0) + 1j*gamma) / den
    return np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))


def voigt(x, A, x0, sigma, gamma):
    return A * voigt_norm(x, x0, sigma, gamma)
