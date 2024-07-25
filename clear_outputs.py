import nbformat
from nbformat.v4.nbbase import new_notebook, new_code_cell

notebook_filename = 'dogs_vs_cats.ipynb'

with open(notebook_filename, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if 'outputs' in cell:
        cell['outputs'] = []
    if 'execution_count' in cell:
        cell['execution_count'] = None

with open(notebook_filename, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Cleared outputs from {notebook_filename}")
