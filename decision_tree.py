# Paula Sophia Santoyo Arteaga
# A01745312
# 04-Sept-2023
# Implementación de una técnica de aprendizaje máquina
# ----------------------------------------------------


# Importar librerias
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Representa un nodo dentro del árbol de decisión
# Almacena la información del indice Gini, la predicción, los nodos 
# hijos (izquierdo y derecho), el atributo y el umbral
class DecisionNode:
    def __init__(self, gini, prediction, left=None, right=None, 
                 attribute=None, threshold=None):
        # Índice gini
        self.gini = gini
        # Predicción
        self.prediction = prediction
        # Nodo izquierdo
        self.left = left
        # Nodo derecho
        self.right = right
        # Atributo a evaluar
        self.attribute = attribute
        # Umbral de decisión
        self.threshold = threshold

def gini(y):
    """
    Función que calcula la impureza del índice gini. Se usa para 
    calcular la probabilidad de que la muestra seleccionada sea 
    clasificada de forma incorrecta    
    """
    classes = np.unique(y)
    total_samples = len(y)
    gini = 1.0

    for c in classes:
        p_c = np.count_nonzero(y == c) / total_samples
        gini -= p_c ** 2

    return gini

def split_data(x, y, attribute, threshold):
    """
    Función que divide el conjunto de datos en 2 conjuntos (x y y) para
    el lado izquierdo y derecho de acuerdo al atributo y umbral
    """
    # El atributo es menor o igual al umbral
    left_side = x[:, attribute] <= threshold
    # El atributo es mayor al umbral
    right_side = x[:, attribute] > threshold
    # Almacena lo que va en el lado izquierdo y derecho
    x_left, y_left = x[left_side], y[left_side]
    x_right, y_right = x[right_side], y[right_side]
    return x_left, y_left, x_right, y_right

def build_tree(x, y, current_depth, max_depth, indent=""):
    """
    Construye el árbol de decisión de forma recursiva con la profundidad
    máxima permitida. Utiliza el índice Gini para elegir el mejor atributo de división
    """
    if current_depth >= max_depth or len(np.unique(y)) == 1:
        predominant_class = np.bincount(y).argmax()
        return DecisionNode(gini=0, prediction=predominant_class)

    # Se inicializan variables
    best_gini = float('inf')
    best_attribute = None
    best_threshold = None

    for attribute in range(x.shape[1]):
        # Se obtienen los valores únicos de los atributos
        unique_values = np.unique(x[:, attribute])
        for threshold in unique_values:
            x_left, y_left, x_right, y_right = split_data(x, y, attribute, threshold)
            if len(y_left) > 0 and len(y_right) > 0:
                # Calcula el índice Gini para el lado izquierdo y derecho
                gini_left = gini(y_left)
                gini_right = gini(y_right)
                # Se calcula el índice Gini ponderado
                weighted_gini = (len(y_left) / len(y)) * gini_left + (len(y_right) / 
                                len(y)) * gini_right
                # Compara el valor del índice Gini para encontrar el mejor y obtener el mejor atributo
                # y umbral
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_attribute = attribute
                    best_threshold = threshold

    if best_attribute is None:
        # Se calcula la clase predominante en caso de no encontrar el mejor atributo para dividir
        predominant_class = np.bincount(y).argmax()
        print(f"{indent}Prediction: {predominant_class}")
        return DecisionNode(gini=0, prediction=predominant_class)

    # Realiza la división en derecha e izquierda de acuerdo con el mejor umbral y atributo encontrados
    x_left, y_left, x_right, y_right = split_data(x, y, best_attribute, best_threshold)
    # Se crea el árbol izquierdo
    left_node = build_tree(x_left, y_left, current_depth + 1, max_depth, indent + "  ")
    # Se crea el árbol derecho
    right_node = build_tree(x_right, y_right, current_depth + 1, max_depth, indent + "  ")
    
    return DecisionNode(gini=best_gini, prediction=None, left=left_node, right=right_node, 
                        attribute=best_attribute, threshold=best_threshold)

def tree_predict(node, sample):
    """
    Función que toma el nodo raíz y una muestra para predecir siguiendo el árbol de decisión    
    """
    if node.prediction is not None:
        return node.prediction
    if sample[node.attribute] <= node.threshold:
        return tree_predict(node.left, sample)
    else:
        return tree_predict(node.right, sample)
    
def load_different_datasets():
    """
    Función que genera diferentes datos de entrenamiento y testing 
    """
    # Datos de entrenamiento aleatorios
    x_train = np.random.rand(100, 10) 
    y_train = np.random.randint(2, size=100)  
    
    # Datos para testing aleatorios
    x_test = np.random.rand(20, 10) 
    y_test = np.random.randint(2, size=20)

    return x_train, y_train, x_test, y_test


# Manda llamar a la función load_different_datasets() para asignarle los valores 
# random a cada una de las 4 variables.Representantan los valores de entrenamiento y testing
x_train, y_train, x_test, y_test = load_different_datasets()


# Divide el dataset de entrenamiento en 2 partes
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


def depth_Tree(depth):
    """
    Función que toma la profundidad máxima como entrada y construye un árbol de decisión con 
    esa profundidad. Realiza predicciones calculando las métricas como precisión, recall y 
    matriz de confusión
    """
    # Determina la profundidad máxima que tendrá el árbol de decisión
    max_depth = depth
    tree = build_tree(x_train, y_train, current_depth=0, max_depth=max_depth)

    # Realizar predicciones en los datos de prueba
    predictions = [tree_predict(tree, sample) for sample in x_test]

    # ---- Calcular métricas de evaluación ----
    
    # Precisión general del modelo
    accuracy = accuracy_score(y_test, predictions) 
    # Mide la proporción de positivos
    precision = np.round(precision_score(y_test, predictions, zero_division=0), 2) 
    # Mide la proporción de positivos que son predichos correctamente
    recall = np.round(recall_score(y_test, predictions), 2) 
    # Almacena la matriz de confusión
    conf_matrix = confusion_matrix(y_test, predictions) 

    # Muestra los resultados
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    
# El código se corre dos veces con profundidades diferentes y datos diferentes
print('========= CORRIDA 1 ==========\n     Maxima profundidad: 3\n')
depth_Tree(3)
print('========= CORRIDA 2 ==========\n     Maxima profundidad: 7\n')
depth_Tree(7)