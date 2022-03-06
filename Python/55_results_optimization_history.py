from typing import Tuple, List
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from evolution.base.base_genome import BaseGenome
from evolution.base.base_strategies import Population
from evolution.base.base_geometry import BaseGeometry
from evolution.camera.camera_algorithm import GeneticCameraAlgorithm
from evolution.camera.camera_genome_factory import CameraGenomeFactory
from evolution.camera.camera_genome_parameters import CameraGenomeParameters
from evolution.camera.camera_rendering import render_geometry_with_camera
from evolution.camera.camera_translator import CameraTranslator
from evolution.camera.object_geometry import ObjGeometry
from evolution.strategies.crossover import SinglePoint, TwoPoint
from evolution.strategies.fitness import DistanceMapWithPunishment, DistanceMap
from evolution.strategies.mutation import BoundedUniformMutation, BoundedDistributionBasedMutation
from evolution.strategies.populate import ValueUniformPopulation
from evolution.strategies.selection import Tournament, RouletteWheel
from evolution.strategies.strategy_bundle import StrategyBundle
from evolution.strategies.termination import NoImprovement


def synthetic_target_edge_image(shape: Tuple[int, int], target_geometry: BaseGeometry,
                                target_genome: BaseGenome) -> np.array:
    edge_image = np.zeros(shape, dtype=np.uint8)
    A, t, r, d = CameraTranslator().translate_genome(target_genome)
    render_geometry_with_camera(edge_image, target_geometry, A, t, r, d, (255,))
    return edge_image


def synthetic_target_dna(shape: Tuple[int, int]):
    fu = max(shape) - 100
    fv = fu
    h, w = shape
    cx, cy = w // 2, h // 2
    tx, ty, tz = 0.00, 2.35, 8.40
    rx, ry, rz = 0.29, 0.00, 0.00
    d0, d1, d2, d3, d4 = np.zeros(5)
    camera_dna = np.array([fu, fv, cx, cy, tx, ty, tz, rx, ry, rz, d0, d1, d2, d3, d4])
    return camera_dna


@dataclass
class HistoryEntry:
    lo_value: float
    mid_value: float
    hi_value: float
    best_value: float

history : List[HistoryEntry] = []
tmp_best = 0

class MyGeneticCameraAlgorithm(GeneticCameraAlgorithm):
    def on_display_population(self, current_generation, population: Population, population_fitness: List[float]):
        super().on_display_population(current_generation, population, population_fitness)
        global history, tmp_best
        best_value = np.max(population_fitness)
        lo_value = np.percentile(population_fitness, 20) #np.min(population_fitness)
        mid_value = np.percentile(population_fitness, 50) # np.mean(population_fitness)
        hi_value = np.percentile(population_fitness, 80) # np.max(population_fitness)
        history.append(HistoryEntry(lo_value, mid_value, hi_value, best_value))


def main():
    np.random.seed(4711)

    DATA_FOLDER = "../data"
    OUTPUT_FOLDER = "../output"
    # 1. Specify all parameters
    image_shape = (image_height, image_width) = 600, 800
    parameters_file = f"{DATA_FOLDER}/squash/squash_parameters.json"
    geometry_file = f"{DATA_FOLDER}/squash/geometries/squash_court.obj"

    # 2. Construct the needed classes and objects for the algorithm
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    genome_parameters = CameraGenomeParameters(parameters_file, image_shape)
    camera_genome_factory = CameraGenomeFactory(genome_parameters)
    camera_translator = CameraTranslator()

    fitting_geometry = ObjGeometry(geometry_file)

    # ######## Create a synthetic, perfect squash court (edge) image ########
    # This step is only needed for synthetic experiments. In a real world application
    # one would extract the edges as binary image of the geometric object instead of
    # rendering the scene! (see real_squash_example)
    real_geometry = fitting_geometry  # Ideally, the target geometry is the fitting geometry
    real_dna = synthetic_target_dna(image_shape)
    real_genome = camera_genome_factory.create(real_dna, "target_camera")

    extracted_edge_image = synthetic_target_edge_image(image_shape, real_geometry, real_genome)
    # "extracted_edge_image" is the binary image which will be used for the algorithm
    # ########################################################################

    # 3. Define your actual strategy
    start_dna = synthetic_target_dna(image_shape)
    random_range = np.array(
        [[-100, -100, -32, -32, -0.1, -1.0, -1.00, np.deg2rad(-20), np.deg2rad(-10), np.deg2rad(-10), -0, -0, -0, -0,
          -3],
         [+100, +100, +32, +32, +0.1, +1.0, +1.00, np.deg2rad(+20), np.deg2rad(+10), np.deg2rad(+10), +0, +0, +0, +0,
          +3]])

    start_dna += np.random.uniform(low=random_range[0], high=random_range[1])

    start_genome = camera_genome_factory.create(start_dna, "opt_camera")

    population_strategy = ValueUniformPopulation(16)
    fitness_strategy = DistanceMapWithPunishment(DistanceMap.DistanceType.L2, .3)
    selection_strategy = RouletteWheel()
    crossover_strategy = SinglePoint()
    mutation_strategy = BoundedDistributionBasedMutation(genome_parameters)
    mutation_strategy.genome_bounds = None
    termination_strategy = NoImprovement(300)

    strategy_bundle = StrategyBundle(population_strategy,
                                     fitness_strategy,
                                     selection_strategy,
                                     crossover_strategy,
                                     mutation_strategy,
                                     termination_strategy)

    # 4. Construct and run the optimization algorithm
    camera_algorithm = MyGeneticCameraAlgorithm(genome_parameters, strategy_bundle, extracted_edge_image, fitting_geometry, headless=False)

    best_possible_fitness = camera_algorithm.fitness(real_genome)
    result = camera_algorithm.run(start_dna)
    best_genome, best_fitness = result.best_genome

    # =========== Present the results ================
    result_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    A_real, t_real, r_real, d_real = camera_translator.translate_genome(real_genome)
    A_start, t_start, r_start, d_start = camera_translator.translate_genome(start_genome)
    A_best, t_best, r_best, d_best = camera_translator.translate_genome(best_genome)

    print(A_start, t_start, r_start, d_start)
    print(A_real, t_real, r_real, d_real)
    print(A_best, t_best, r_best, d_best)

    render_geometry_with_camera(result_image, real_geometry, A_real, t_real, r_real, d_real, (0, 200, 0), 8)
    render_geometry_with_camera(result_image, fitting_geometry, A_start, t_start, r_start, d_start, (255, 0, 0), 2)
    render_geometry_with_camera(result_image, fitting_geometry, A_best, t_best, r_best, d_best, (0, 0, 255), 2)

    cv.imshow("I", result_image)
    cv.waitKey(0)

    return history, best_possible_fitness

def plot_history(best_values, lo_values, mid_values, hi_values):
    tax = np.arange(len(best_values))
    plt.figure(figsize=(16, 9))

    plt.fill_between(x=tax, y1=lo_values, y2=hi_values, alpha=0.25)
    plt.plot(tax, best_values, lw=1)
    plt.plot(tax, mid_values, lw=1)

    plt.xlim(0, len(best_values)-300)
    plt.ylim(0)
    plt.show()  

if __name__ == '__main__':
#    history, best_possible_fitness = main()
#    best_values = np.array(best_possible_fitness-[h.best_value for h in history])
#    lo_values = np.array(best_possible_fitness-[h.lo_value for h in history])
#    mid_values = np.array(best_possible_fitness-[h.mid_value for h in history])
#    hi_values = np.array(best_possible_fitness-[h.hi_value for h in history])
#    np.savez("temporary.npz", best_values=best_values, lo_values=lo_values, mid_values=mid_values, hi_values=hi_values)

    import os
    cwd = os.getcwd()
    os.chdir(cwd + "/Python/")
    npzfile = np.load("./temporary.npz")
    best_values = npzfile["best_values"]
    lo_values = npzfile["lo_values"]
    mid_values = npzfile["mid_values"]
    hi_values = npzfile["hi_values"]

    plot_history(best_values, lo_values, mid_values, hi_values)

