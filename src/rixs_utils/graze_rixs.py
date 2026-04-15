from .helper_functions import binned_spectrum
from .helper_functions import apply_custom_plot_style, apply_custom_plot_style_light
from .helper_functions import line, parabola,  gaussian

from numba import set_num_threads
import os
import sys
import copy

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit, minimize_scalar
from scipy.ndimage import  distance_transform_edt, gaussian_filter1d
from scipy.interpolate import interp1d
from typing import  Union
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


from .helper_functions import (line, gaussian, parabola, build_lookup,
                               apply_curve_correction, get_pgm_en_graze,
                               read_scan_xy, find_scan_files, rotate_events_xy,
                               prepare_scans, _rot_to_local,_solve_y_branches,
                               _lower_upper_branch, intersection_with_mask)



def calibration(
    scans_path,
    calib_scan_nums,
    scan_ranges,
    spec_dir="../SpecData",
    use_scan_energies=True,
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
        use_scan_energies=use_scan_energies,
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

    # --- Maskenerstellung: Parabel oder Linie als untere Grenze ---
    nx, ny = histogram_all.shape
    xg = np.arange(nx, dtype=float)
    yg = np.arange(ny, dtype=float)
    Xg, Yg = np.meshgrid(xg, yg, indexing="ij")
    circle1_inside = (Xg - x0) ** 2 + (Yg - y0) ** 2 <= r**2
    circle2_inside = (Xg - x0_2) ** 2 + (Yg - y0_2) ** 2 <= r**2

    if bottom_bound > 0:
        # Linie als untere Grenze, rotiert um den Winkel
        # Die Linie ist y = bottom_bound im unrotierten System
        # Nach Rotation: y' = ...
        # Wir drehen die horizontale Linie y=bottom_bound um -angle um (900,900)
        theta = np.deg2rad(-angle)
        y_line = np.full_like(xg, bottom_bound, dtype=float)
        x_rot = (xg - 900.0) * np.cos(theta) - (y_line - 900.0) * np.sin(theta) + 900.0
        y_rot = (xg - 900.0) * np.sin(theta) + (y_line - 900.0) * np.cos(theta) + 900.0
        # Für jedes xg: untere Grenze ist y_rot
        line_halfspace = Yg >= y_rot[:, None]
        mask_region_raw = circle1_inside & circle2_inside & line_halfspace
        dist_to_boundary = distance_transform_edt(mask_region_raw)
        mask_region = mask_region_raw & (dist_to_boundary >= buffer_edges) # type:ignore
        mask_region_plot = np.ma.masked_where(~mask_region, mask_region.astype(float))
        # Dummy-Variablen für Parabel-Modus
        y_parabola = None
        pts_fit = None
    else:
        # Parabel-Modus wie bisher
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
        # --- Fitpunktgrenzen anpassen: ---
        if bottom_bound > 0:
            # untere Grenze ist die Linie (nach Rotation)
            # y_rot = (reg["x_bottom"] - 900.0) * np.sin(theta) + (bottom_bound - 900.0) * np.cos(theta) + 900.0
            theta = np.deg2rad(-angle)
            y_line_rot = (reg["x_bottom"] - 900.0) * np.sin(theta) + (bottom_bound - 900.0) * np.cos(theta) + 900.0
            valid_y_raw = np.arange(int(y_line_rot), reg["top_jump"] + 1)
        else:
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
        axes_1[1].set_title("Calibration Lines - Rotated, masked Detector Image")
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
        if bottom_bound > 0:
            # Linie plotten
            theta_rot = np.deg2rad(-angle)
            y_line = np.full_like(xg, bottom_bound, dtype=float)
            x_rot = (xg - 900.0) * np.cos(theta_rot) - (y_line - 900.0) * np.sin(theta_rot) + 900.0
            y_rot = (xg - 900.0) * np.sin(theta_rot) + (y_line - 900.0) * np.cos(theta_rot) + 900.0
            axes_1[1].plot(x_rot, y_rot, color="orange", lw=2.5, label="untere Maskenlinie")
        else:
            # Parabel und Maskenpunkte plotten
            axes_1[1].scatter(
                pts_fit[:, 0],#type: ignore
                pts_fit[:, 1],#type: ignore
                marker="P",
                s=120,
                c="darkviolet",
                label="Bottom-Punkte"
            )
            axes_1[1].plot(
                x_line, y1, #type: ignore
                color="darkviolet",
                lw=2.5,
                label="Gedrehte Parabel (Ast 1)"
            )
            axes_1[1].plot(
                x_line, y2,#type: ignore
                color="deepskyblue",
                lw=2.0,
                alpha=0.9,
                label="Gedrehte Parabel (Ast 2)"
            )
        # Fitpunkte und deren Parabelfits immer plotten
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
            norm_factor = np.max(np.sum(hist[:, min_y:max_y], axis=1)[l_lim:r_lim])
            ax[1].plot( # type:ignore
                np.arange(hist.shape[0])[l_lim:r_lim],
                np.sum(hist[:, min_y:max_y], axis=1)[l_lim:r_lim]/norm_factor,
                ".",
                color="black",
            )
            ax[1].plot( # type:ignore
                x_vals[l_lim:r_lim],
                gaussian(x_vals, *popt)[l_lim:r_lim]/norm_factor,
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


def process_RIXS(
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
        raise ValueError("artifacts_region must have the same length as scan_ranges.")

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
            print(f"No files found for scan range {measurement}.")
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

            if artifacts_region[num] is not None:
                for region in artifacts_region[num]:
                    y_start, y_end = region

                    ax[0].axhspan(
                        max(y_start, min_y),
                        min(y_end, max_y),
                        color='red',
                        alpha=0.3,   # Transparenz
                        zorder=10    # über dem Bild
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