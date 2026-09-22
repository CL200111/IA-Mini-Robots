# Informe Técnico y Académico: Codificador de Display de 7 Segmentos mediante Programación Genética

**Asignatura:** Computación Evolutiva / Inteligencia Artificial  
**Proyecto:** Diseño Automatizado de Circuitos Lógicos BCD a 7 Segmentos  
**Librería Principal:** DEAP (*Distributed Evolutionary Algorithms in Python*)  
**Autor:** Repositorio GitHub de Programación Genética  

---

## 📄 Resumen Ejecutivo

El presente informe constituye la documentación técnica y académica completa para el diseño automatizado de un **codificador de display de 7 segmentos** mediante **Programación Genética (PG)** [1, 16]. El sistema evoluciona dinámicamente un conjunto de **7 ecuaciones booleanas estructuradas en forma de árbol** (un circuito lógico por cada segmento: `a`, `b`, `c`, `d`, `e`, `f`, `g`) a partir de una entrada en Código Decimal Codificado en Binario (BCD de 4 bits: $A, B, C, D$) [1, 3]. 

El proyecto resuelve un problema clásico de diseño en ingeniería digital sin recurrir a la simplificación manual por mapas de Karnaugh ni a algoritmos heurísticos deterministas [19, 41]. Se utiliza la librería **DEAP** para la ejecución evolutiva y **Matplotlib** para la renderización gráfica del display [2, 3].

---

## 1. 🧬 Marco Teórico Formal de la Programación Genética

La **Programación Genética (PG)** es una disciplina de la computación evolutiva en la que los **programas de computador evolucionan de forma automática** para resolver problemas específicos sin ser programados explícitamente [16, 19]. A diferencia de los Algoritmos Genéticos (AG) tradicionales que operan sobre cadenas lineales fijas (vectores o listas), la PG opera sobre **estructuras de datos dinámicas, no lineales y en forma de árbol** [16, 17].

```
                 AND (Nodo Raíz - Función)
                /   \
              OR     NOT (Nodos Internos - Funciones)
             /  \     |
            A    B    C  (Hojas - Terminales)
```

### Características Fundamentales del Genotipo en PG:
1. **Material Genético No Lineal**: Representado mediante **árboles de sintaxis abstracta (AST)** [17, 30]. Los nodos internos representan **funciones o compuertas lógicas** ($F$) con una aridad específica, mientras que los nodos hoja representan **terminales o entradas** ($T$) [20, 30].
2. **Longitud Variable**: Las estructuras pueden crecer o contraerse a lo largo de las generaciones según las necesidades del problema [17].
3. **Material Genético Ejecutable (Equivalencia Genotipo-Fenotipo)**: El árbol generado se compila o interpreta directamente como una función ejecutable para calcular su aptitud frente al entorno [17, 23].
4. **Preservación Sintáctica**: Los operadores genéticos de cruce y mutación están diseñados para intercambiar subárboles completos, garantizando que los hijos sean siempre expresiones sintácticamente válidas [18, 28].

---

## 2. 🎯 Los 6 Pasos Preparatorios de Koza / Martínez

De acuerdo con la metodología clásica expuesta por John Koza y el profesor José J. Martínez P., la aplicación rigurosa de la PG requiere la definición explícita de **6 pasos preparatorios** [19, 20]:

| Paso Preparatorio | Definición en el Proyecto | Justificación Técnica |
| :--- | :--- | :--- |
| **1. El Problema** [20] | Síntesis lógica de un codificador BCD a 7 segmentos [1]. | Convertir 4 bits de entrada BCD ($A, B, C, D$) en la activación correcta (0 o 1) para cada uno de los 7 segmentos (`a` a `g`) representando los dígitos 0 al 9 [3]. |
| **2. Conjunto de Terminales ($T$)** [20] | $T = \{A, B, C, D, 0, 1\}$ [3, 5] | Las variables $A, B, C, D$ son los 4 bits BCD ($A$ es el MSB y $D$ es el LSB) [3, 8]. Se incluyen las constantes booleanas `0` y `1` [3, 5]. |
| **3. Conjunto de Funciones Primitivas ($F$)** [20] | $F = \{\text{AND}_2, \text{OR}_2, \text{XOR}_2, \text{NAND}_2, \text{NOR}_2, \text{NOT}_1\}$ [3, 5] | Un conjunto completo de compuertas lógicas digitales [3, 5]. Cumple la **Propiedad de Clausura**, permitiendo que cualquier función acepte las salidas de las demás o cualquier terminal [5, 21]. |
| **4. Medida de Aptitud ($f_{apt}$)** [20, 22] | $f_{apt} = \frac{1.0}{0.1 + \sum_{j=0}^{9} \|y_j - \hat{y}_{j}\|}$ [7, 8, 46] | Función de aptitud inversamente proporcional al error absoluto acumulado en los 10 dígitos [8, 44, 46]. Acotada en un valor máximo de $10.0$ cuando el error es 0 [8, 46]. |
| **5. Parámetros de Control** [20, 23] | Tamaño de Población $N=200$, Generaciones $G=40$, Probabilidad de Cruce $p_c=0.7$, Probabilidad de Mutación $p_m=0.2$, Torneo $k=3$ [8, 10] | Configuración equilibrada que garantiza la exploración estocástica y la convergencia en pocos minutos sin caer en estancamiento [8, 10, 14]. |
| **6. Criterio de Terminación y Resultado** [20, 23] | Alcanzar $G=40$ generaciones o una aptitud $f_{apt} = 10.0$ [8, 10]. | Se extrae el mejor individuo guardado en el *Hall of Fame* (Salón de la Fama) por cada segmento [10]. |

---

## 3. 🔍 Desglose y Análisis Técnico del Código Python

A continuación se analiza en detalle la estructura y funcionamiento del script `codificador_7_segmentos.py`.

### 3.1. Mapeo BCD y Tabla de Verdad
```python
TABLA_7SEG = {
    0: {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 0},
    1: {'a': 0, 'b': 1, 'c': 1, 'd': 0, 'e': 0, 'f': 0, 'g': 0},
    ...
    9: {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 0, 'f': 1, 'g': 1},
}
```
* **Explicación**: Especifica el estado deseado de encendido ($1$) o apagado ($0$) para cada segmento dado un dígito decimal del $0$ al $9$ [3, 4]. Funciona como el conjunto de entrenamiento de la PG [4].

### 3.2. Configuración del PrimitiveSet
```python
pset = gp.PrimitiveSet("MAIN", 4)
pset.renameArguments(ARG0='A', ARG1='B', ARG2='C', ARG3='D')

def _and(x, y): return int(x) & int(y)
def _not(x): return 1 - int(x)

pset.addPrimitive(_and, 2, name="AND")
...
pset.addTerminal(1, name="1")
pset.addTerminal(0, name="0")
```
* **Explicación**: Se define el dominio funcional [5]. Las compuertas booleanas se implementan con operadores a nivel de bits (`&`, `|`, `^`) sobre valores binarios binarizados [3, 5]. Se renombran las variables de entrada a $A, B, C, D$ para una lectura limpia de las ecuaciones [3, 5].

### 3.3. Evaluación de la Función de Aptitud
```python
def evaluar_segmento(individual, segmento, toolbox_compile):
    func = toolbox_compile(expr=individual)
    error_acumulado = 0.0
    for digito in range(10):
        bit_A = (digito >> 3) & 1
        bit_B = (digito >> 2) & 1
        bit_C = (digito >> 1) & 1
        bit_D = digito & 1
        try:
            val = func(bit_A, bit_B, bit_C, bit_D)
            salida = 1 if float(val) >= 0.5 else 0
        except Exception:
            salida = 0
        esperado = TABLA_7SEG[digito][segmento]
        error_acumulado += abs(salida - esperado)
    return (1.0 / (0.1 + error_acumulado),)
```
* **Conversión BCD mediante desplazamientos de bits**: Para obtener cada bit BCD del dígito en tiempo real:
  * `(digito >> 3) & 1`: Extrae el bit más significativo (MSB, Bit $A$).
  * `(digito >> 2) & 1`: Extrae el Bit $B$.
  * `(digito >> 1) & 1`: Extrae el Bit $C$.
  * `digito & 1`: Extrae el bit menos significativo (LSB, Bit $D$).
* **Acumulación de error**: Se calcula la diferencia absoluta entre la respuesta del árbol y la respuesta esperada [7, 8]. Si el árbol produce respuestas perfectas para los 10 dígitos ($error = 0$), el denominador es $0.1$ y la aptitud resultante es $1.0 / 0.1 = 10.0$ [8, 46].

### 3.4. Evolución Modular de los 7 Segmentos
```python
def evolucionar_segmento(segmento, generaciones=40, tam_poblacion=200):
    fit_class_name = f"FitnessMax_{segmento}"
    ind_class_name = f"Individual_{segmento}"
    ...
```
* **Aislamiento dinámico de clases**: La librería DEAP registra tipos de datos globales en el módulo `creator`. Al evolucionar 7 árboles distintos en el mismo script, se crean clases personalizadas como `Individual_a`, `Individual_b`, etc., evitando la sobrescritura accidental de aptitudes entre segmentos [9].
* **Operadores Evolutivos**:
  * `genHalfAndHalf`: Generación inicial combinada de árboles completos y de crecimiento aleatorio con límites de profundidad $min=1, max=4$ [6, 7].
  * `cxOnePoint`: Cruce de subárboles en un punto elegido al azar [10, 25].
  * `mutUniform`: Mutación uniforme que reemplaza un nodo o subárbol [10, 25].
  * `selTournament(tournsize=3)`: Selección por torneo entre 3 individuos aleatorios [10].

---

## 4. 🎓 Análisis Académico Avanzado y Justificaciones de Diseño

### 4.1. Demostración Matemática de la Función de Aptitud
En problemas de optimización simbólica y síntesis lógica, minimizar el error absoluto suma $\sum_{j=0}^{M-1} |y_j - \hat{y}_j|$ puede provocar indeterminaciones cuando el error es cero ($\frac{1}{0} \to \infty$) [44, 45]. Para solucionar esto, se adopta la formulación propuesta por Martínez (2026) [46]:

$$f_{apt}(i) = \frac{1}{\epsilon + \sum_{j=0}^{9} |y_j - \hat{y}_{i,j}|}$$

Donde $\epsilon = 0.1$ es una constante de suavizado [8, 46]. 
* Si $\sum |y_j - \hat{y}_j| = 0 \implies f_{apt} = \frac{1}{0.1} = 10.0$ (Solución perfecta) [8, 46].
* Si el error es máximo ($\sum |y_j - \hat{y}_j| = 10$) $\implies f_{apt} = \frac{1}{10.1} \approx 0.099$.

Esta escala acotada en $[0.099, 10.0]$ estabiliza la selección por torneo y evita explosiones numéricas [8, 10].

### 4.2. Descomposición Modular: ¿Por qué 7 Árboles Independientes?
En lugar de forzar a la PG a evolucionar un único árbol multisalida de gran complejidad, el problema se divide en **7 subproblemas monosalida independientes** [14].
* **Reducción del espacio de búsqueda**: Un árbol de salida múltiple requiere definir tipos complejos y estructuras entrelazadas. Al descomponerlo, el espacio de búsqueda para cada segmento se reduce a $2^{2^4} = 2^{16} = 65,536$ funciones booleanas posibles, lo que permite a la PG converger al circuito óptimo en cuestión de segundos [8, 14].

### 4.3. Control del Crecimiento Excesivo (*Bloat*)
El fenómeno de *bloat* en PG consiste en el crecimiento desmedido del tamaño de los árboles sin una mejora proporcional en la aptitud [14, 29, 48]. En nuestro script:
* Se limita la profundidad inicial mediante `gp.genHalfAndHalf(..., min_=1, max_=4)` [6, 14].
* Esto mantiene las expresiones booleanas simplificadas y ejecutables con bajo consumo de memoria y alta interpretabilidad [14].

---

## 5. 🚀 Guía de Ejecución y Repositorio GitHub

### Requisitos Previos e Instalación
Para ejecutar el proyecto en cualquier entorno local o en GitHub Codespaces:

```bash
pip install deap matplotlib numpy
```

### Ejecución del Script
```bash
python codificador_7_segmentos.py
```

### Garantía de Reproducibilidad
El código incluye semillas estocásticas explícitas (`SEMILLA = 42`) para `random` y `numpy` [14]. Al ejecutar el script, la PG convergerá de manera idéntica hacia los mismos árboles y valores de aptitud perfecta ($10.0000$) en todas las corridas, facilitando la evaluación del docente [14].

---

## 📚 Referencias Bibliográficas

1. **Martínez P., José J.** (2026). *Conceptos Básicos de Programación Genética*. Universidad Nacional de Colombia. [16]
2. **Koza, John R.** (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press. [54]
3. **Fortin, F. A., De Rainville, F. M., Gardner, M. A., Parizeau, M., & Gagné, C.** (2012). *DEAP: Evolutionary Algorithms Made Easy*. Journal of Machine Learning Research, 13, 2171-2175.
4. **Guía de Implementación**: *Codificador de 7 segmentos con Programación Genética y DEAP*. [1]
