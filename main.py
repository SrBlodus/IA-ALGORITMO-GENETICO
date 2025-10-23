import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class Individuo:
    """Representa un individuo en la población."""
    
    def __init__(self, genes: List[float] = None, rango: Tuple[float, float] = (-10, 10)):
        """
        Inicializa un individuo.
        
        Args:
            genes: Lista con los valores de los genes (cromosoma)
            rango: Tupla con el rango de valores permitidos
        """
        if genes is None:
            # Representación real: un gen que representa el valor de x
            self.genes = [random.uniform(rango[0], rango[1])]
        else:
            self.genes = genes
        
        self.fitness = 0
        self.rango = rango
    
    def __repr__(self):
        return f"Individuo(x={self.genes[0]:.4f}, fitness={self.fitness:.4f})"


class AlgoritmoGenetico:
    """Implementa un algoritmo genético con selección por ranking lineal."""
    
    def __init__(self, 
                 tam_poblacion: int = 50,
                 prob_mutacion: float = 0.1,
                 prob_cruce: float = 0.8,
                 max_generaciones: int = 200,
                 rango: Tuple[float, float] = (-10, 10),
                 tolerancia_convergencia: float = 1e-6):
        """
        Inicializa el algoritmo genético.
        
        Args:
            tam_poblacion: Tamaño de la población
            prob_mutacion: Probabilidad de mutación
            prob_cruce: Probabilidad de cruce
            max_generaciones: Número máximo de generaciones
            rango: Rango de valores para x
            tolerancia_convergencia: Tolerancia para considerar convergencia
        """
        self.tam_poblacion = tam_poblacion
        self.prob_mutacion = prob_mutacion
        self.prob_cruce = prob_cruce
        self.max_generaciones = max_generaciones
        self.rango = rango
        self.tolerancia_convergencia = tolerancia_convergencia
        
        self.poblacion: List[Individuo] = []
        self.mejor_individuo: Individuo = None
        self.historial_fitness: List[float] = []
        self.historial_promedio: List[float] = []
        self.generacion_actual = 0
        
    def funcion_objetivo(self, x: float) -> float:
        """
        Función objetivo a minimizar: f(x) = |x³ - 20x + 5|
        Representa la pérdida energética de una bomba hidráulica.
        
        Args:
            x: Valor de la variable
            
        Returns:
            Valor de la función objetivo
        """
        return abs(x**3 - 20*x + 5)
    
    def inicializar_poblacion(self):
        """Crea la población inicial con individuos aleatorios."""
        self.poblacion = [Individuo(rango=self.rango) 
                         for _ in range(self.tam_poblacion)]
        self.evaluar()
    
    def evaluar(self):
        """Evalúa el fitness de todos los individuos de la población."""
        for individuo in self.poblacion:
            x = individuo.genes[0]
            # Como queremos minimizar, el fitness es el negativo de la función
            # O podemos usar 1/(1+f(x)) para tener valores positivos
            valor_funcion = self.funcion_objetivo(x)
            individuo.fitness = 1 / (1 + valor_funcion)
        
        # Actualizar mejor individuo
        mejor_actual = max(self.poblacion, key=lambda ind: ind.fitness)
        if self.mejor_individuo is None or mejor_actual.fitness > self.mejor_individuo.fitness:
            self.mejor_individuo = Individuo(genes=mejor_actual.genes[:])
            self.mejor_individuo.fitness = mejor_actual.fitness
    
    def seleccionar_ranking_lineal(self) -> List[Individuo]:
        """
        Selección escalada por ranking lineal.
        
        Los individuos se ordenan por fitness y se les asigna una probabilidad
        de selección proporcional a su posición en el ranking.
        
        Returns:
            Lista de individuos seleccionados para reproducción
        """
        # Ordenar población por fitness (de menor a mayor)
        poblacion_ordenada = sorted(self.poblacion, key=lambda ind: ind.fitness)
        
        # Asignar ranking (1 al peor, tam_poblacion al mejor)
        # Probabilidad lineal: P(i) = (2-s)/N + 2*i*(s-1)/(N*(N-1))
        # donde s es el parámetro de presión selectiva (1 <= s <= 2)
        s = 1.5  # Presión selectiva moderada
        N = self.tam_poblacion
        
        # Calcular probabilidades basadas en ranking
        probabilidades = []
        for i in range(N):
            rank = i + 1  # Ranking de 1 a N
            prob = (2 - s) / N + 2 * rank * (s - 1) / (N * (N - 1))
            probabilidades.append(prob)
        
        # Normalizar probabilidades
        suma_prob = sum(probabilidades)
        probabilidades = [p / suma_prob for p in probabilidades]
        
        # Seleccionar individuos usando ruleta con las probabilidades calculadas
        seleccionados = []
        for _ in range(self.tam_poblacion):
            r = random.random()
            acumulado = 0
            for i, prob in enumerate(probabilidades):
                acumulado += prob
                if r <= acumulado:
                    # Crear copia del individuo seleccionado
                    ind_seleccionado = Individuo(genes=poblacion_ordenada[i].genes[:])
                    ind_seleccionado.fitness = poblacion_ordenada[i].fitness
                    seleccionados.append(ind_seleccionado)
                    break
        
        return seleccionados
    
    def cruzar(self, padre1: Individuo, padre2: Individuo) -> Tuple[Individuo, Individuo]:
        """
        Realiza cruce aritmético entre dos padres.
        
        Args:
            padre1: Primer padre
            padre2: Segundo padre
            
        Returns:
            Tupla con dos hijos
        """
        if random.random() < self.prob_cruce:
            # Cruce aritmético: hijo = alpha*padre1 + (1-alpha)*padre2
            alpha = random.random()
            
            hijo1_genes = [alpha * padre1.genes[0] + (1 - alpha) * padre2.genes[0]]
            hijo2_genes = [(1 - alpha) * padre1.genes[0] + alpha * padre2.genes[0]]
            
            hijo1 = Individuo(genes=hijo1_genes, rango=self.rango)
            hijo2 = Individuo(genes=hijo2_genes, rango=self.rango)
        else:
            # No hay cruce, se copian los padres
            hijo1 = Individuo(genes=padre1.genes[:], rango=self.rango)
            hijo2 = Individuo(genes=padre2.genes[:], rango=self.rango)
        
        return hijo1, hijo2
    
    def mutar(self, individuo: Individuo):
        """
        Aplica mutación uniforme al individuo.
        
        La mutación uniforme reemplaza el gen con un valor aleatorio
        dentro del rango permitido.
        
        Args:
            individuo: Individuo a mutar
        """
        if random.random() < self.prob_mutacion:
            # Mutación uniforme: reemplazar con valor aleatorio en el rango
            individuo.genes[0] = random.uniform(self.rango[0], self.rango[1])
    
    def verificar_convergencia(self) -> bool:
        """
        Verifica si el algoritmo ha convergido.
        
        Returns:
            True si ha convergido, False en caso contrario
        """
        if len(self.historial_fitness) < 10:
            return False
        
        # Verificar si los últimos 10 valores son similares
        ultimos_valores = self.historial_fitness[-10:]
        variacion = max(ultimos_valores) - min(ultimos_valores)
        
        return variacion < self.tolerancia_convergencia
    
    def ejecutar(self) -> Tuple[Individuo, List[float], List[float]]:
        """
        Ejecuta el algoritmo genético.
        
        Returns:
            Tupla con (mejor_individuo, historial_fitness, historial_promedio)
        """
        print("="*60)
        print("ALGORITMO GENÉTICO - OPTIMIZACIÓN DE BOMBA HIDRÁULICA")
        print("="*60)
        print(f"Función objetivo: f(x) = |x³ - 20x + 5|")
        print(f"Estrategia: Selección por Ranking Lineal")
        print(f"Mutación: Uniforme")
        print(f"\nParámetros:")
        print(f"  - Tamaño de población: {self.tam_poblacion}")
        print(f"  - Probabilidad de cruce: {self.prob_cruce}")
        print(f"  - Probabilidad de mutación: {self.prob_mutacion}")
        print(f"  - Generaciones máximas: {self.max_generaciones}")
        print(f"  - Rango de búsqueda: {self.rango}")
        print("="*60)
        
        # Inicializar población
        self.inicializar_poblacion()
        
        # Registrar estadísticas iniciales
        fitness_inicial = [ind.fitness for ind in self.poblacion]
        self.historial_fitness.append(self.mejor_individuo.fitness)
        self.historial_promedio.append(np.mean(fitness_inicial))
        
        print(f"\nGeneración 0:")
        print(f"  Mejor: x={self.mejor_individuo.genes[0]:.6f}, "
              f"f(x)={self.funcion_objetivo(self.mejor_individuo.genes[0]):.6f}")
        
        # Loop principal
        for generacion in range(1, self.max_generaciones + 1):
            self.generacion_actual = generacion
            
            # Selección
            seleccionados = self.seleccionar_ranking_lineal()
            
            # Cruce y mutación
            nueva_poblacion = []
            for i in range(0, self.tam_poblacion, 2):
                padre1 = seleccionados[i]
                padre2 = seleccionados[i + 1] if i + 1 < self.tam_poblacion else seleccionados[0]
                
                hijo1, hijo2 = self.cruzar(padre1, padre2)
                self.mutar(hijo1)
                self.mutar(hijo2)
                
                nueva_poblacion.extend([hijo1, hijo2])
            
            # Actualizar población
            self.poblacion = nueva_poblacion[:self.tam_poblacion]
            
            # Evaluar nueva población
            self.evaluar()
            
            # Registrar estadísticas
            fitness_actual = [ind.fitness for ind in self.poblacion]
            self.historial_fitness.append(self.mejor_individuo.fitness)
            self.historial_promedio.append(np.mean(fitness_actual))
            
            # Mostrar progreso cada 20 generaciones
            if generacion % 20 == 0:
                print(f"\nGeneración {generacion}:")
                print(f"  Mejor: x={self.mejor_individuo.genes[0]:.6f}, "
                      f"f(x)={self.funcion_objetivo(self.mejor_individuo.genes[0]):.6f}")
                print(f"  Fitness promedio: {np.mean(fitness_actual):.6f}")
            
            # Verificar convergencia
            if self.verificar_convergencia():
                print(f"\n{'='*60}")
                print(f"¡Convergencia alcanzada en la generación {generacion}!")
                print(f"{'='*60}")
                break
        
        # Resultados finales
        print(f"\n{'='*60}")
        print("RESULTADOS FINALES")
        print(f"{'='*60}")
        print(f"Generaciones ejecutadas: {self.generacion_actual}")
        print(f"Mejor solución encontrada:")
        print(f"  x = {self.mejor_individuo.genes[0]:.8f}")
        print(f"  f(x) = {self.funcion_objetivo(self.mejor_individuo.genes[0]):.8f}")
        print(f"  Fitness = {self.mejor_individuo.fitness:.8f}")
        print(f"{'='*60}")
        
        return self.mejor_individuo, self.historial_fitness, self.historial_promedio
    
    def graficar_resultados(self):
        """Genera gráficos de los resultados del algoritmo."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gráfico 1: Evolución del fitness
        axes[0, 0].plot(self.historial_fitness, 'b-', linewidth=2, label='Mejor fitness')
        axes[0, 0].plot(self.historial_promedio, 'r--', linewidth=1.5, label='Fitness promedio')
        axes[0, 0].set_xlabel('Generación')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].set_title('Evolución del Fitness')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gráfico 2: Función objetivo
        x_vals = np.linspace(self.rango[0], self.rango[1], 1000)
        y_vals = [self.funcion_objetivo(x) for x in x_vals]
        
        axes[0, 1].plot(x_vals, y_vals, 'g-', linewidth=2)
        axes[0, 1].axvline(x=self.mejor_individuo.genes[0], color='r', 
                          linestyle='--', linewidth=2, label='Mejor solución')
        axes[0, 1].scatter([self.mejor_individuo.genes[0]], 
                          [self.funcion_objetivo(self.mejor_individuo.genes[0])],
                          color='r', s=100, zorder=5)
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('f(x)')
        axes[0, 1].set_title('Función Objetivo: f(x) = |x³ - 20x + 5|')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gráfico 3: Distribución de la población final
        fitness_final = [ind.fitness for ind in self.poblacion]
        axes[1, 0].hist(fitness_final, bins=20, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(x=self.mejor_individuo.fitness, color='r', 
                          linestyle='--', linewidth=2, label='Mejor fitness')
        axes[1, 0].set_xlabel('Fitness')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].set_title('Distribución del Fitness (Población Final)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gráfico 4: Valores de x en la población final
        x_final = [ind.genes[0] for ind in self.poblacion]
        axes[1, 1].hist(x_final, bins=20, color='lightcoral', edgecolor='black')
        axes[1, 1].axvline(x=self.mejor_individuo.genes[0], color='r', 
                          linestyle='--', linewidth=2, label='Mejor x')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_title('Distribución de x (Población Final)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Código principal
if __name__ == "__main__":
    # Crear instancia del algoritmo genético
    ag = AlgoritmoGenetico(
        tam_poblacion=50,
        prob_mutacion=0.1,
        prob_cruce=0.8,
        max_generaciones=200,
        rango=(-10, 10),
        tolerancia_convergencia=1e-6
    )
    
    # Ejecutar el algoritmo
    mejor, historial_fitness, historial_promedio = ag.ejecutar()
    
    # Graficar resultados
    ag.graficar_resultados()
    
    # Información adicional
    print("\n" + "="*60)
    print("ANÁLISIS ADICIONAL")
    print("="*60)
    print(f"\nIntegrantes del Grupo 4:")
    print("  - Alejandro Benítez")
    print("  - Nicolás Espinoza")
    print(f"\nMétodo de selección: Ranking Lineal (Escalada)")
    print("  - Ventaja: Evita convergencia prematura")
    print("  - Presión selectiva moderada (s=1.5)")
    print(f"\nTipo de representación: Real (valores continuos)")
    print(f"Tipo de cruce: Aritmético")
    print(f"Tipo de mutación: Uniforme")
    print("="*60)