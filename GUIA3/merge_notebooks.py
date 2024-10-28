import nbformat

def merge_notebooks(notebooks, output_file):
    merged = nbformat.v4.new_notebook()
    for notebook in notebooks:
        with open(notebook, 'r') as f:
            nb = nbformat.read(f, as_version=4)
            for cell in nb.cells:
                merged.cells.append(cell)
    with open(output_file, 'w') as f:
        nbformat.write(merged, f)
    print(f"Notebooks merged into {output_file}")


if __name__ == '__main__':
    notebooks = [f"ejercicio{i}.ipynb" for i in range(1, 5)]
    output_file = ('informe_GUIA3.ipynb')
    merge_notebooks(notebooks, output_file)
