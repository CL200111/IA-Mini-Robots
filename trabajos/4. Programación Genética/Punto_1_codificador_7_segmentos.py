"""
================================================================================
CODIFICADOR DE 7 SEGMENTOS BCD MEDIANTE PROGRAMACIÓN GENÉTICA (DEAP)
================================================================================
Autor: Trabajo Académico de Programación Genética
Descripción:
    Este script utiliza Programación Genética (PG) con la librería DEAP para
    evolucionar un conjunto de 7 circuitos lógicos en forma de árbol (uno por
    cada segmento: a, b, c, d, e, f, g). Los circuitos mapean entradas BCD de
    4 bits (A, B, C, D) a la activación de los segmentos de un display de 7
    segmentos para los dígitos 0 al 9.

    Incluye una interfaz gráfica interactiva con Matplotlib para visualizar
    los dígitos representados en el display.
================================================================================
"""

import random
import operator
from typing import Dict, Tuple, Any, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from deap import base, creator, tools, gp, algorithms

# ------------------------------------------------------------------------------
# 0. CONFIGURACIÓN DE SEMILLAS PARA REPRODUCIBILIDAD
# ------------------------------------------------------------------------------
SEMILLA = 42
random.seed(SEMILLA)
np.random.seed(SEMILLA)

# ------------------------------------------------------------------------------
# 1. TABLA DE VERDAD DEL DISPLAY DE 7 SEGMENTOS
# ------------------------------------------------------------------------------
# Mapeo BCD (0-9) a los 7 segmentos del display:
# a: arriba, b: sup-der, c: inf-der, d: abajo, e: inf-izq, f: sup-izq, g: centro
TABLA_7SEG: Dict[int, Dict[str, int]] = {
    0: {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 0},
    1: {'a': 0, 'b': 1, 'c': 1, 'd': 0, 'e': 0, 'f': 0, 'g': 0},
    2: {'a': 1, 'b': 1, 'c': 0, 'd': 1, 'e': 1, 'f': 0, 'g': 1},
    3: {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 0, 'f': 0, 'g': 1},
    4: {'a': 0, 'b': 1, 'c': 1, 'd': 0, 'e': 0, 'f': 1, 'g': 1},
    5: {'a': 1, 'b': 0, 'c': 1, 'd': 1, 'e': 0, 'f': 1, 'g': 1},
    6: {'a': 1, 'b': 0, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1},
    7: {'a': 1, 'b': 1, 'c': 1, 'd': 0, 'e': 0, 'f': 0, 'g': 0},
    8: {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1},
    9: {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 0, 'f': 1, 'g': 1},
}

SEGMENTOS: List[str] = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

# ------------------------------------------------------------------------------
# 2. CONFIGURACIÓN DEL CONJUNTO PRIMITIVO (FUNCIONES Y TERMINALES)
# ------------------------------------------------------------------------------
# 4 Entradas independientes BCD: A (MSB), B, C, D (LSB)
pset = gp.PrimitiveSet("MAIN", 4)
pset.renameArguments(ARG0='A', ARG1='B', ARG2='C', ARG3='D')

# Definición de compuertas lógicas mediante operadores bitwise seguros
def _and(x: int, y: int) -> int:
    return int(x) & int(y)

def _or(x: int, y: int) -> int:
    return int(x) | int(y)

def _xor(x: int, y: int) -> int:
    return int(x) ^ int(y)

def _nand(x: int, y: int) -> int:
    return 1 - (int(x) & int(y))

def _nor(x: int, y: int) -> int:
    return 1 - (int(x) | int(y))

def _not(x: int) -> int:
    return 1 - int(x)

# Incorporación de funciones primitivas al PrimitiveSet
pset.addPrimitive(_and, 2, name="AND")
pset.addPrimitive(_or, 2, name="OR")
pset.addPrimitive(_xor, 2, name="XOR")
pset.addPrimitive(_nand, 2, name="NAND")
pset.addPrimitive(_nor, 2, name="NOR")
pset.addPrimitive(_not, 1, name="NOT")

# Terminales constantes booleanas
pset.addTerminal(1, name="1")
pset.addTerminal(0, name="0")

# ------------------------------------------------------------------------------
# 3. FUNCIÓN DE EVALUACIÓN DE APTITUD (FITNESS)
# ------------------------------------------------------------------------------
def evaluar_segmento(individual: gp.PrimitiveTree, segmento: str, toolbox_compile: Any) -> Tuple[float]:
    """
    Evalúa un árbol individual de PG para un segmento específico.
    Compara la respuesta lógica del árbol contra la tabla de verdad oficial.
    
    Aptitud = 1.0 / (0.1 + error_absoluto_acumulado)
    Aptitud máxima = 10.0 (cuando error_absoluto_acumulado == 0).
    """
    func = toolbox_compile(expr=individual)
    error_acumulado = 0.0
    
    for digito in range(10):
        # Conversión del dígito (0-9) a sus 4 bits BCD (A, B, C, D)
        bit_A = (digito >> 3) & 1
        bit_B = (digito >> 2) & 1
        bit_C = (digito >> 1) & 1
        bit_D = digito & 1
        
        try:
            val = func(bit_A, bit_B, bit_C, bit_D)
            # Binarización del resultado
            salida = 1 if float(val) >= 0.5 else 0
        except Exception:
            # En caso de error de ejecución inesperado, la respuesta se penaliza como 0
            salida = 0
            
        esperado = TABLA_7SEG[digito][segmento]
        error_acumulado += abs(salida - esperado)
        
    fitness_val = 1.0 / (0.1 + error_acumulado)
    return (fitness_val,)

# ------------------------------------------------------------------------------
# 4. FUNCIÓN PARA EVOLUCIONAR EL CIRCUITO DE UN SEGMENTO
# ------------------------------------------------------------------------------
def evolucionar_segmento(segmento: str, generaciones: int = 40, tam_poblacion: int = 200) -> Tuple[gp.PrimitiveTree, Any]:
    """
    Crea un entorno de evolución aislado en DEAP para un segmento particular
    evitando colisiones de clases dinámicas y optimizando el espacio de búsqueda.
    """
    fit_class_name = f"FitnessMax_{segmento}"
    ind_class_name = f"Individual_{segmento}"
    
    if not hasattr(creator, fit_class_name):
        creator.create(fit_class_name, base.Fitness, weights=(1.0,))
    if not hasattr(creator, ind_class_name):
        creator.create(ind_class_name, gp.PrimitiveTree, fitness=getattr(creator, fit_class_name))

    tb = base.Toolbox()
    # Generación inicial genHalfAndHalf con restricción de profundidad (min_=1, max_=4) para mitigar el bloat
    tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    tb.register("individual", tools.initIterate, getattr(creator, ind_class_name), tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("compile", gp.compile, pset=pset)

    def eval_fn(ind):
        return evaluar_segmento(ind, segmento, tb.compile)

    tb.register("evaluate", eval_fn)
    # Operadores evolutivos
    tb.register("mate", gp.cxOnePoint)  # Cruce de un punto en subárboles
    tb.register("mutate", gp.mutUniform, expr=tb.expr, pset=pset)  # Mutación uniforme
    tb.register("select", tools.selTournament, tournsize=3)  # Selección por torneo

    poblacion = tb.population(n=tam_poblacion)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)

    # Ejecución del algoritmo evolutivo simple
    algorithms.eaSimple(
        poblacion, tb, cxpb=0.7, mutpb=0.2,
        ngen=generaciones, stats=stats,
        halloffame=hof, verbose=False
    )

    mejor_individuo = hof[0]
    funcion_compilada = tb.compile(expr=mejor_individuo)
    return mejor_individuo, funcion_compilada

# ------------------------------------------------------------------------------
# 5. VISUALIZACIÓN GRÁFICA DEL DISPLAY
# ------------------------------------------------------------------------------
def dibujar_display(digito: int, salida: Dict[str, int]) -> None:
    """
    Dibuja el display de 7 segmentos utilizando Matplotlib.
    Los segmentos activos se iluminan en rojo y los inactivos en gris claro.
    """
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    grosor = 0.25
    largo = 1.2

    # Geometría de los 7 segmentos (x, y, ancho, alto)
    coords = {
        'a': (-largo / 2, largo / 2, largo, grosor),
        'b': (largo / 2 - grosor / 2, largo / 2 - largo + grosor / 2, grosor, largo),
        'c': (largo / 2 - grosor / 2, -largo / 2 - largo / 2 + grosor / 2, grosor, largo),
        'd': (-largo / 2, -largo - grosor / 2, largo, grosor),
        'e': (-largo / 2 - grosor / 2, -largo / 2 - largo / 2 + grosor / 2, grosor, largo),
        'f': (-largo / 2 - grosor / 2, largo / 2 - largo + grosor / 2, grosor, largo),
        'g': (-largo / 2, -grosor / 2, largo, grosor),
    }

    for seg, (x, y, w, h) in coords.items():
        encendido = salida.get(seg, 0) == 1
        color = 'red' if encendido else '#E0E0E0'
        rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)

    ax.set_title(f"Dígito BCD Evaluado: {digito}", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# 6. PROGRAMA PRINCIPAL
# ------------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" PROGRAMACIÓN GENÉTICA: CODIFICADOR DE 7 SEGMENTOS BCD")
    print(" Evolucionando circuitos lógicos con la librería DEAP...")
    print("=" * 70)

    arboles_evolucionados: Dict[str, gp.PrimitiveTree] = {}
    funciones_compiladas: Dict[str, Any] = {}

    # Evolucionar los 7 segmentos individualmente
    for seg in SEGMENTOS:
        print(f"  --> Entrenando segmento '{seg}'...", end=" ", flush=True)
        mejor, func = evolucionar_segmento(seg, generaciones=40, tam_poblacion=200)
        arboles_evolucionados[seg] = mejor
        funciones_compiladas[seg] = func
        aptitud = mejor.fitness.values[0]
        print(f"Completado | Aptitud final = {aptitud:.4f} (Max: 10.0)")

    print("\n" + "=" * 70)
    print(" ¡ENTRENAMIENTO FINALIZADO CON ÉXITO!")
    print("=" * 70 + "\n")

    # Mostrar expresiones lógicas obtenidas
    print("Ecuaciones Booleanas Evolucionadas (Estructura Árbol):")
    for seg in SEGMENTOS:
        print(f"  Segmento {seg}: {arboles_evolucionados[seg]}")
    print("\n" + "-" * 70)

    # Bucle interactivo de prueba
    while True:
        print("\nOpciones: [0-9] para probar un dígito | 'salir' para finalizar")
        entrada = input("Ingrese una opción: ").strip()

        if entrada.lower() == 'salir':
            print("Finalizando ejecución. ¡Gracias!")
            break

        if not entrada.isdigit() or not (0 <= int(entrada) <= 9):
            print("⚠️  Entrada no válida. Ingrese un entero entre 0 y 9.")
            continue

        digito = int(entrada)
        bit_A = (digito >> 3) & 1
        bit_B = (digito >> 2) & 1
        bit_C = (digito >> 1) & 1
        bit_D = digito & 1

        salida_pg = {}
        for seg, func in funciones_compiladas.items():
            val = func(bit_A, bit_B, bit_C, bit_D)
            salida_pg[seg] = 1 if float(val) >= 0.5 else 0

        esperado = TABLA_7SEG[digito]
        
        print(f"\n[Dígito {digito}] -> Entrada BCD (A,B,C,D): ({bit_A}, {bit_B}, {bit_C}, {bit_D})")
        print(f"  Salida de PG:        {salida_pg}")
        print(f"  Salida Esperada:     {esperado}")

        if salida_pg == esperado:
            print("  Status: ✅ COINCIDENCIA PERFECTA")
        else:
            print("  Status: ⚠️ DIFERENCIA DETECTADA")

        # Visualizar display
        dibujar_display(digito, salida_pg)

if __name__ == "__main__":
    main()
