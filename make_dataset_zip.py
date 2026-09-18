"""Package pec2 source into pec2_dataset.zip for Kaggle upload."""
import os
import zipfile

EXCLUDE_DIRS = {".pytest_cache", "__pycache__", "results_ci", "results_smoke",
                "results_full", "kaggle_out", "kaggle_pull", "kaggle_stage"}
EXCLUDE_FILES = {"run_ci.txt", "make_dataset_zip.py", "build_notebook.py",
                 "parse_klog.py", "pec2_dataset.zip"}

with zipfile.ZipFile("pec2_dataset.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f in EXCLUDE_FILES or not f.endswith((".py", ".md")):
                continue
            p = os.path.join(root, f)
            arc = os.path.relpath(p, ".").replace(os.sep, "/")
            zf.write(p, arc)

print("pec2_dataset.zip contents:")
for n in sorted(zipfile.ZipFile("pec2_dataset.zip").namelist()):
    print(" ", n)
