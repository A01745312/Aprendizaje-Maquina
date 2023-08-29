# Paula Sophia Santoyo Arteaga
# A01745312
# 28-Ago-2023
# Implementación de una técnica de aprendizaje máquina
# ----------------------------------------------------

# Importar librerias
import numpy as np

# Representa un nodo dentro del árbol de decisión
class DecisionNode:
    def __init__(self, gini, prediction, left=None, right=None, 
                 attribute=None, threshold=None): 
        self.gini = gini
        self.prediction = prediction
        self.left = left
        self.right = right
        self.attribute = attribute
        self.threshold = threshold

# Función que calcula el índice Gini para medir la impureza de las muestras
def gini(y):
    classes = np.unique(y)
    total_samples = len(y)
    gini = 1.0

    for c in classes:
        p_c = np.count_nonzero(y == c) / total_samples
        gini -= p_c ** 2

    return gini

# Función que divide los datos en izquierda y derecha de acuerdo al attribute y threshold
def split_data(x, y, attribute, threshold):
    left_side = x[:, attribute] <= threshold
    right_side = x[:, attribute] > threshold
    x_left, y_left = x[left_side], y[left_side]
    x_right, y_right = x[right_side], y[right_side]
    return x_left, y_left, x_right, y_right

# Función que construye el árbol de decisión 
def build_tree(x, y, current_depth, max_depth, indent=""):
    # Se decide si se debe detener la construcción del árbol o continua
    if current_depth >= max_depth or len(np.unique(y)) == 1:
        predominant_class = np.bincount(y).argmax()
        return DecisionNode(gini=0, prediction=predominant_class)

    best_gini = float('inf')
    best_attribute = None
    best_threshold = None

    # Busca encontrar el mejor attribute y threshold para dividir los datos en el árbol
    for attribute in range(x.shape[1]):
        unique_values = np.unique(x[:, attribute])
        for threshold in unique_values:
            x_left, y_left, x_right, y_right = split_data(x, y, attribute, threshold)
            if len(y_left) > 0 and len(y_right) > 0:
                gini_left = gini(y_left)
                gini_right = gini(y_right)
                weighted_gini = (len(y_left) / len(y)) * gini_left + (len(y_right) / 
                                len(y)) * gini_right
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_attribute = attribute
                    best_threshold = threshold

    # Si no se encuentra una división óptima se crea un nodo hoja para predecir la clase predominante
    if best_attribute is None:
        predominant_class = np.bincount(y).argmax()
        print(f"{indent}Prediction: {predominant_class}")
        return DecisionNode(gini=0, prediction=predominant_class)

    # Se construyen los nodos izquierdo y derecho del árbol de decisión 
    x_left, y_left, x_right, y_right = split_data(x, y, best_attribute, best_threshold)
    left_node = build_tree(x_left, y_left, current_depth + 1, max_depth, indent + "  ")
    right_node = build_tree(x_right, y_right, current_depth + 1, max_depth, indent + "  ")
    
    return DecisionNode(gini=best_gini, prediction=None, left=left_node, right=right_node, 
                        attribute=best_attribute, threshold=best_threshold)

# Función que realiza predicciones a partir de una muestra
def tree_predict(node, sample):
    if node.prediction is not None:
        return node.prediction
    if sample[node.attribute] <= node.threshold:
        return tree_predict(node.left, sample)
    else:
        return tree_predict(node.right, sample)

# --------------------------------------------------------------------------

# SECCIÓN DE DATOS A OCUPAR

# Profundidad máxima (niveles desde el nodo raíz)
max_depth = 5

# Datos de entrenamiento
x_train = np.array([
    [5, 2],
    [3, 8],
    [8, 6],
    [1, 4],
    [9, 1]
])

y_train = np.array([0, 1, 1, 0, 1])

# Datos para testing
x_test = np.array([
    [4, 3],
    [7, 5],
    [2, 4],
    [8, 2],
    [6, 7],
    [1, 5]
])

# -----------------------------------------------------------------------

tree = build_tree(x_train, y_train, current_depth=0, max_depth=max_depth)

for sample in x_test:
    prediction = tree_predict(tree, sample)
    print(f"Para la muestra {sample}, la prediccion es: {prediction}")
