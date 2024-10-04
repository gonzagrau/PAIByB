import os

# Define la carpeta base

carpeta_base = 'PIByB_4'
lista_paths =[]
# Recorre todos los archivos dentro de la carpeta
for root, dirs, files in os.walk(carpeta_base):
    for file in files:
        # Obtiene el path absoluto del archivo
        path_absoluto = os.path.join(root, file)
       
        lista_paths.append(path_absoluto)
        # Obtiene el path relativo con respecto a la carpeta base
        path_relativo = os.path.relpath(path_absoluto, carpeta_base)
      
for i in lista_paths:
    print(i)