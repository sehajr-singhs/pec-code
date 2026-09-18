"""Rebuild kaggle_pec2.ipynb as a fully self-contained notebook.

The pec2 source (all modules + tests) is embedded as a base64 zip in the
first cell, so the Kaggle kernel needs no dataset attachment and no internet.
"""
import base64
import json
import zipfile

with zipfile.ZipFile("pec2_dataset.zip") as zf:
    blob = base64.b64encode(zf.read("pec2/__init__.py") and open("pec2_dataset.zip", "rb").read()).decode()

    # split into ~60k chunks to keep cell sources manageable
    chunks = [blob[i:i + 60000] for i in range(0, len(blob), 60000)]

setup_lines = [
    "import base64, os, sys, zipfile\n",
    "\n",
    "WORK = \"/kaggle/working\"\n",
    "CODE = os.path.join(WORK, \"code\")\n",
    "os.makedirs(CODE, exist_ok=True)\n",
    f"B64 = {''.join(repr(c + ('' if i == len(chunks) - 1 else '')) for i, c in enumerate([]))}",
]
# build the B64 assignment from chunk parts explicitly
parts_src = []
for i, c in enumerate(chunks):
    parts_src.append(f"    {c!r}" + (" +\n" if i < len(chunks) - 1 else "\n"))
setup_src = [
    "import base64, os, sys, zipfile\n",
    "\n",
    "WORK = \"/kaggle/working\"\n",
    "CODE = os.path.join(WORK, \"code\")\n",
    "os.makedirs(CODE, exist_ok=True)\n",
    "B64 = (\n",
    *parts_src,
    ")\n",
    "with zipfile.ZipFile(__import__('io').BytesIO(base64.b64decode(B64))) as zf:\n",
    "    zf.extractall(CODE)\n",
    "sys.path.insert(0, CODE)\n",
    "print(\"code root:\", CODE)\n",
    "print(\"contents:\", sorted(os.listdir(CODE))[:12])\n",
]

old = json.load(open("kaggle_pec2.ipynb"))
new_cells = []
replaced = False
for cell in old["cells"]:
    if cell["cell_type"] == "code" and not replaced:
        # the first code cell is always the setup cell -> embed the bundle here
        new_cells.append({"cell_type": "code", "metadata": {}, "outputs": [],
                          "execution_count": None, "source": setup_src})
        replaced = True
        continue
    new_cells.append(cell)

assert replaced, "no code cell found to replace"
json.dump({"cells": new_cells, "metadata": old["metadata"],
           "nbformat": old["nbformat"], "nbformat_minor": old["nbformat_minor"]},
          open("kaggle_pec2.ipynb", "w"), indent=1)
print("notebook rebuilt:", len(chunks), "chunks,", len(blob), "b64 chars")
