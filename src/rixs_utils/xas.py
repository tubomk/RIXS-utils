from pathlib import Path
import numpy as np

def load_scan(scan_number, spec_folder):
    spec_folder = Path(spec_folder)

    scan_block = None
    spec_file_used = None

    # search all spec files
    for spec_file in spec_folder.glob("*.spec"):
        content = spec_file.read_text(encoding="utf-8")
        scans = content.split("#S ")

        for block in scans[1:]:
            if int(block.splitlines()[0].split()[0]) == scan_number:
                scan_block = block
                spec_file_used = spec_file
                break

        if scan_block is not None:
            break

    if scan_block is None:
        raise ValueError(f"Scan {scan_number} not found in any spec file in {spec_folder}")

    lines = scan_block.splitlines()
    header = None
    data_rows = []
    read_data = False

    for line in lines:
        if line.startswith("#L "):
            header = line.split()[1:]
            read_data = True
            continue
        if not read_data:
            continue
        if not line.strip():
            continue
        if line.startswith("#"):
            break
        data_rows.append([float(v) for v in line.split()])

    if header is None or not data_rows:
        raise ValueError(f"No tabular data found for scan {scan_number}")

    arr = np.array(data_rows)
    idx = {name: i for i, name in enumerate(header)}

    required = ["pgm_en", "kth1", "kth2", "kth3"]
    missing = [name for name in required if name not in idx]
    if missing:
        raise KeyError(f"Missing columns in scan {scan_number}: {missing}")

    pgm_en = arr[:, idx["pgm_en"]]
    kth1 = arr[:, idx["kth1"]]
    kth2 = arr[:, idx["kth2"]]
    kth3 = arr[:, idx["kth3"]]

    return pgm_en, kth1, kth2, kth3
