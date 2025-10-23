# 🧬 Algoritmo Genético - Optimización de Bomba Hidráulica

**Grupo 4**: Alejandro Benítez y Nicolás Espinoza

## 📋 Descripción del Problema

Este proyecto implementa un **algoritmo genético** para optimizar el funcionamiento de una bomba hidráulica, minimizando la pérdida energética representada por la función:

```
f(x) = |x³ - 20x + 5|
```

Donde `x` representa un parámetro operacional de la bomba (por ejemplo, velocidad de rotación, presión, o caudal) y `f(x)` representa la **pérdida energética** que queremos minimizar.

## 🎯 Objetivo

Encontrar el valor óptimo de `x` que **minimice** la función `f(x)`, reduciendo así la pérdida energética de la bomba hidráulica y mejorando su eficiencia.

## 🔧 Estrategia de Selección: Ranking Lineal (Escalada)

La **selección por ranking lineal** es una técnica que:

1. **Ordena** a todos los individuos de la población según su fitness
2. **Asigna probabilidades** de selección basadas en la posición (ranking) y no en el valor absoluto del fitness
3. **Evita convergencia prematura** dando oportunidades a individuos menos aptos
4. **Mantiene diversidad** en la población

### Fórmula de Probabilidad

```
P(i) = (2-s)/N + 2*i*(s-1)/(N*(N-1))
```

Donde:
- `i` = posición en el ranking (1 = peor, N = mejor)
- `N` = tamaño de la población
- `s` = parámetro de presión selectiva (1.5 en nuestra implementación)

### Ventajas de Ranking Lineal

✅ **Robustez**: No afectado por valores extremos de fitness  
✅ **Balance**: Equilibrio entre exploración y explotación  
✅ **Diversidad**: Mantiene variabilidad genética  
✅ **Control**: Ajustable mediante el parámetro `s`

## ⚙️ Parámetros del Algoritmo

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **Tamaño de población** | 50 | Número de individuos por generación |
| **Probabilidad de cruce** | 0.8 (80%) | Probabilidad de reproducción entre padres |
| **Probabilidad de mutación** | 0.1 (10%) | Probabilidad de mutación uniforme |
| **Rango de búsqueda** | [-10, 10] | Espacio de búsqueda para `x` |
| **Generaciones máximas** | 200 | Criterio de parada |
| **Tolerancia convergencia** | 1e-6 | Diferencia mínima para considerar convergencia |
| **Tipo de cruce** | Aritmético | Combinación lineal de genes |
| **Tipo de mutación** | Uniforme | Reemplazo aleatorio en el rango |
| **Representación** | Real | Valores continuos (no binarios) |

## 🚀 Instalación y Ejecución

### 1. Requisitos previos

- Python 3.6 o superior
- pip (gestor de paquetes de Python)

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

O manualmente:

```bash
pip install numpy matplotlib
```

### 3. Ejecutar el algoritmo

```bash
python algoritmo_genetico.py
```

## 📊 Resultados y Visualización

El programa genera automáticamente:

### Salida en consola:
- Parámetros del algoritmo
- Progreso por generación (cada 20 generaciones)
- Mejor solución encontrada
- Valor mínimo de pérdida energética

### Gráficos generados:

1. **Evolución del Fitness**: Muestra cómo mejora el mejor fitness y el promedio a lo largo de las generaciones
2. **Función Objetivo**: Visualiza f(x) y marca la solución óptima encontrada
3. **Distribución del Fitness Final**: Histograma de fitness en la última población
4. **Distribución de x Final**: Histograma de valores x en la última población

## 📈 Análisis de Resultados

### Convergencia
El algoritmo se detiene cuando:
- Alcanza 200 generaciones, O
- Los últimos 10 valores de fitness varían menos de 1e-6

### Interpretación Física
- **x óptimo**: Representa el parámetro operacional ideal de la bomba
- **f(x) mínimo**: Pérdida energética mínima alcanzable
- Valores más bajos de f(x) = mayor eficiencia energética

## 🧬 Características del Algoritmo

### Representación del Cromosoma
- **Tipo**: Real (valores continuos)
- **Genes**: Un solo gen que representa el valor de `x`

### Operadores Genéticos

**Selección**: Ranking Lineal
- Presión selectiva moderada
- Previene convergencia prematura

**Cruce**: Aritmético
```python
hijo1 = α * padre1 + (1-α) * padre2
hijo2 = (1-α) * padre1 + α * padre2
```

**Mutación**: Uniforme
```python
gen_mutado = random.uniform(rango_min, rango_max)
```

## 📁 Estructura del Código

```python
class Individuo:
    - genes: Lista con el valor de x
    - fitness: Valor de aptitud (inverso de f(x))

class AlgoritmoGenetico:
    - inicializar_poblacion()
    - evaluar()
    - seleccionar_ranking_lineal()  ← Estrategia del Grupo 4
    - cruzar()
    - mutar()
    - verificar_convergencia()
    - ejecutar()
    - graficar_resultados()
```

## 🔬 Fundamento Matemático

### Función de Fitness
Como queremos **minimizar** f(x), transformamos a maximización:

```
fitness = 1 / (1 + f(x))
```

Así:
- f(x) pequeño → fitness grande (bueno)
- f(x) grande → fitness pequeño (malo)

### Análisis de la Función Objetivo

La función `f(x) = |x³ - 20x + 5|` tiene:
- **Mínimos locales** en el rango [-10, 10]
- **Comportamiento cúbico** con término lineal
- **Valor absoluto** que la hace siempre positiva

## 📝 Notas Importantes

1. **Mutación Uniforme**: Elegida según especificaciones del trabajo
2. **Convergencia**: El algoritmo puede converger antes de 200 generaciones
3. **Aleatoriedad**: Diferentes ejecuciones pueden dar resultados ligeramente distintos
4. **Semilla**: Se puede fijar `random.seed()` para reproducibilidad

## 🎓 Referencias Teóricas

- **Selección por Ranking**: Baker, J.E. (1985). Adaptive selection methods for genetic algorithms
- **Algoritmos Genéticos**: Goldberg, D.E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning
- **Mutación Uniforme**: Mühlenbein, H. & Schlierkamp-Voosen, D. (1993)

## 👥 Autores

- **Alejandro Benítez**
- **Nicolás Espinoza**

**Grupo 4** - Selección Escalada (Ranking Lineal)

---

## 📧 Contacto

Para dudas o consultas sobre la implementación, contactar a los autores del Grupo 4.

---

**Última actualización**: Octubre 2025  
**Versión**: 1.0