import nbformat
import os

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


def main():
    base_dir = './'
    base_notebook = 'README_intro_TP4.ipynb'
    notebooks = [base_notebook]
    notebooks.extend(sorted([path for path in os.listdir(base_dir)
                              if path.endswith('.ipynb')
                              and 'Ej' in path]))
    print(notebooks)
    input('Press Enter to merge the notebooks')
    output_file = ('informe_guia4_BAJLEC_NEIRA_GRAU.ipynb')
    merge_notebooks(notebooks, output_file)


if __name__ == '__main__':
    main()