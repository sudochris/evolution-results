from abc import ABC, abstractmethod
from typing import Tuple, List
from dataclasses import dataclass
import os 

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from evolution.base.base_genome import BaseGenome
from evolution.base.base_geometry import BaseGeometry, PlaneGeometry, DenseGeometry
from evolution.base.base_strategies import Population
from evolution.camera.camera_algorithm import GeneticCameraAlgorithm
from evolution.camera.camera_genome_factory import CameraGenomeFactory
from evolution.camera.camera_genome_parameters import CameraGenomeParameters
from evolution.camera.camera_rendering import render_geometry_with_camera, project_points
from evolution.camera.camera_translator import CameraTranslator
from evolution.camera.object_geometry import ObjGeometry
from evolution.strategies.crossover import SinglePoint
from evolution.strategies.fitness import DistanceMapWithPunishment, DistanceMap
from evolution.strategies.mutation import BoundedDistributionBasedMutation
from evolution.strategies.populate import ValueUniformPopulation
from evolution.strategies.selection import RouletteWheel
from evolution.strategies.strategy_bundle import StrategyBundle
from evolution.strategies.termination import NoImprovement

current_bgr_image = None

os.chdir(os.getcwd() + "/SubmissionFigures")

full_geometry = ObjGeometry("data/squash/geometries/full_squash_court.obj")
floor_geometry = ObjGeometry("data/squash/geometries/squash_court_floor.obj")

ground_poly_world = floor_geometry.world_points

WINDOWS = {
    "LIVE": "LiveWindow",
    "RESULT": "ResultWindow",
    "CONTROLS": "ControlwsWindow"
}

def create_windows():
    for window_name in WINDOWS.values():
        cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)

def destroy_windows():
    cv.destroyAllWindows()

def create_overlay(image, A, t, r, d) -> np.array:
    overlay = np.zeros_like(image)

    image_points = project_points(ground_poly_world, A, t, r, d)
    image_points = image_points.reshape((-1, 2))
    image_height, image_width = image.shape[:2]
    poly_line_points = np.clip(image_points, (0, 0), (image_width, image_height))
    pts = np.array([poly_line_points], dtype=np.int64)
    cv.fillPoly(overlay, pts, (128, 128, 128), cv.LINE_AA)
    cv.polylines(overlay, pts , True, (255, 255, 255), 2, cv.LINE_AA)
    return overlay

def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]
    return color_range.reshape(256, 1, 3)

class MyGeneticCameraAlgorithm(GeneticCameraAlgorithm):
    def on_best_genome_found(self, new_best: BaseGenome, genome_fitness: float):
        super().on_best_genome_found(new_best, genome_fitness)

        if self._current_best_genome is not None:
            bgr_display = current_bgr_image.copy()
    
            A_best, t_best, r_best, d_best = self.translator.translate_genome(self._current_best_genome)
            overlay = create_overlay(bgr_display, A_best, t_best, r_best, d_best)
            bgr_display = cv.addWeighted(bgr_display, 1.0, overlay, 0.3, 0)

            cv.imshow(WINDOWS["LIVE"], bgr_display)
            cv.waitKey(1)


class EdgeStrategy(ABC):
    @abstractmethod
    def extract_edge(self, image: np.ndarray) -> np.ndarray:
        pass

class CannyExtractor(EdgeStrategy):
    def __init__(self, threshold1: int, threshold2: int):
        self._threshold1 = threshold1
        self._threshold2 = threshold2

    def extract_edge(self, image: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(gray, self._threshold1, self._threshold2)
        return canny


@dataclass
class ImageConfiguration:
    _identifier: str
    _folder: str
    _extension: str
    _is_synth: bool
    _edge_strategy: EdgeStrategy
    _start_camera_params: tuple
    _image_filename : str = "Court"
    _image_blurred_filename : str= "CourtBlurred"
    _geometry_filename : str = "Geometry"
    
    def get_file_in_folder(self, filename: str):
        return f"{self._folder}/{filename}"

    @property
    def identifier(self):
        return self._identifier

    @property
    def folder(self):
        return self._folder

    @property
    def image(self):
        return cv.imread(self.get_file_in_folder(f"{self._image_filename}.{self._extension}"), cv.IMREAD_COLOR)

    @property
    def image_blur(self):
        return cv.imread(self.get_file_in_folder(f"{self._image_blurred_filename}.{self._extension}"), cv.IMREAD_COLOR)

    @property
    def edge_image(self):
        return self._edge_strategy.extract_edge(self.image)

    @property
    def geometry(self):
        return ObjGeometry(self.get_file_in_folder(f"{self._geometry_filename}.obj"))

    @property
    def image_shape(self):
        return self.image.shape[:2]

    @property
    def start_camera(self):
        _tx, _ty, _tz, _rx, _ry, _rz = self._start_camera_params
        return _construct_camera_array(self.image_shape, _tx, _ty, _tz, _rx, _ry, _rz)

    @property
    def hhsv_range(self):
        return self._hhsv_color_range


def _construct_camera_array_full(image_shape, fu:float, fv:float, tx: float, ty: float, tz: float, rx: float, ry: float, rz: float):
    (image_height, image_width) = image_shape

    cx, cy = image_width // 2, image_height // 2
    d0, d1, d2, d3, d4 = np.zeros(5)

    return np.array([fu, fv, cx, cy, tx, ty, tz, rx, ry, rz, d0, d1, d2, d3, d4])


def _construct_camera_array(image_shape, tx: float, ty: float, tz: float, rx: float, ry: float, rz: float):
    fu = fv = max(image_shape) - 100
    return _construct_camera_array_full(image_shape, fu, fv, tx, ty, tz, rx, ry, rz)


def evolve(image_configuration: ImageConfiguration):
    edge_image = image_configuration.edge_image

    geometry = image_configuration.geometry
    start_dna = image_configuration.start_camera
    image_shape = (image_height, image_width) = image_configuration.image_shape

    global current_bgr_image

    current_bgr_image = image_configuration.image_blur

    parameters_file = "data/squash/squash_parameters.json"

    genome_parameters = CameraGenomeParameters(parameters_file, image_shape)
    genome_factory = CameraGenomeFactory(genome_parameters)
    population_strategy = ValueUniformPopulation(16)
    fitness_strategy = DistanceMapWithPunishment(DistanceMap.DistanceType.L2, .3)
    mutation_strategy = BoundedDistributionBasedMutation(genome_parameters)
    selection_strategy = RouletteWheel()
    crossover_strategy = SinglePoint()

    mutation_strategy.genome_bounds = None
    
    termination_strategy = NoImprovement(300)

    strategy_bundle = StrategyBundle(population_strategy,
                                     fitness_strategy,
                                     selection_strategy,
                                     crossover_strategy,
                                     mutation_strategy,
                                     termination_strategy)

    camera_algorithm = MyGeneticCameraAlgorithm(genome_parameters, strategy_bundle,
                                                edge_image, geometry, headless=True)

    result = camera_algorithm.run(start_dna)
    best_genome, _ = result.best_genome

    translator = CameraTranslator()
    bgr_display = image_configuration.image_blur.copy()
    start_genome = genome_factory.create(start_dna, "start_dna")

    A_start, t_start, r_start, d_start = translator.translate_genome(start_genome)
    A_best, t_best, r_best, d_best = translator.translate_genome(best_genome)

    start_overlay = create_overlay(bgr_display, A_start, t_start, r_start, d_start)
    best_overlay = create_overlay(bgr_display, A_best, t_best, r_best, d_best)

    start_display = bgr_display.copy()
    best_display = bgr_display.copy()

    fitness_display = cv.normalize(camera_algorithm._fitness_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    fitness_display = cv.applyColorMap(fitness_display, get_mpl_colormap("RdYlBu"))

    start_display = cv.addWeighted(start_display, 0.8, start_overlay, 0.8, 0)
    best_display = cv.addWeighted(best_display, 0.8, best_overlay, 0.8, 0)

    render_geometry_with_camera(start_display, full_geometry, A_start, t_start, r_start, d_start, (0, 0, 128), 8, line_type=cv.LINE_AA)
    render_geometry_with_camera(best_display, full_geometry, A_best, t_best, r_best, d_best, (0, 96, 0), 2, line_type=cv.LINE_AA)
    render_geometry_with_camera(best_display, geometry, A_best, t_best, r_best, d_best, (0, 255, 0), 4, line_type=cv.LINE_AA)

    top_row = cv.hconcat([start_display, best_display])
    bottom_row = cv.hconcat([fitness_display, fitness_display])
    full_image = cv.vconcat([top_row, bottom_row])
    cv.imshow(WINDOWS["RESULT"], full_image)
    cv.waitKey(0)

    cv.imwrite(f"{image_configuration.folder}/TMP_Result_start.png", start_display)
    cv.imwrite(f"{image_configuration.folder}/TMP_Result_fitness.png", fitness_display)
    cv.imwrite(f"{image_configuration.folder}/TMP_Result_best.png", best_display)


class DataProvider(ABC):
    @abstractmethod
    def define_experiments(self) -> List[ImageConfiguration]:
        pass

class ExperimentData(DataProvider):
    def define_experiments(self) -> List[ImageConfiguration]:
        start_camera = (0.00, 2.35, 8.40, 0.29, 0.00, 0.00)

        synthA = ImageConfiguration("SynthA", "data/Courts/SynthA", "png", True, CannyExtractor(0, 158), start_camera)
        synthB = ImageConfiguration("SynthB", "data/Courts/SynthB", "png", True, CannyExtractor(0, 158), start_camera)
        synthC = ImageConfiguration("SynthC", "data/Courts/SynthC", "png", True, CannyExtractor(0, 158), start_camera)
        synthD = ImageConfiguration("SynthD", "data/Courts/SynthD", "png", True, CannyExtractor(0, 158), start_camera)
        courtA = ImageConfiguration("CourtA", "data/Courts/CourtA", "png", False, CannyExtractor(0, 255), start_camera)
        courtB = ImageConfiguration("CourtB", "data/Courts/CourtB", "png", False, CannyExtractor(0, 255), start_camera)
        courtC = ImageConfiguration("CourtC", "data/Courts/CourtC", "png", False, CannyExtractor(57, 119), start_camera)
        courtD = ImageConfiguration("CourtD", "data/Courts/CourtD", "png", False, CannyExtractor(50, 230), start_camera)
        courtE = ImageConfiguration("CourtE", "data/Courts/CourtE", "png", False, CannyExtractor(0, 255), start_camera)
        courtF = ImageConfiguration("CourtF", "data/Courts/CourtF", "png", False, CannyExtractor(0, 255), start_camera)

        return [synthA, synthB, synthC, synthD, courtA, courtB, courtC, courtD, courtE, courtF]

def find_canny_params(image_configuration: ImageConfiguration):
    
    noop = lambda i : i
    running = True
    
    court_image = image_configuration.image
    edge_strategy = image_configuration._edge_strategy
    
    THRESH1_KEY, THRESH2_KEY = "Threshold1", "Threshold2"

    threshold1 = edge_strategy._threshold1
    threshold2 = edge_strategy._threshold2

    fitness_strategy = DistanceMapWithPunishment(DistanceMap.DistanceType.L2, .3)
    
    cv.createTrackbar(THRESH1_KEY, WINDOWS["CONTROLS"], threshold1, 255, noop)
    cv.createTrackbar(THRESH2_KEY, WINDOWS["CONTROLS"], threshold2, 255, noop)

    while running:
        threshold1 = cv.getTrackbarPos(THRESH1_KEY, WINDOWS["CONTROLS"])
        threshold2 = cv.getTrackbarPos(THRESH2_KEY, WINDOWS["CONTROLS"])
        
        edge_strategy._threshold1 = threshold1
        edge_strategy._threshold2 = threshold2

        edge_image = edge_strategy.extract_edge(court_image)
        fitness_map = fitness_strategy.create_fitness(edge_image)

        fitness_display = cv.normalize(fitness_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        fitness_display_color = cv.applyColorMap(fitness_display, get_mpl_colormap("RdYlBu"))

        result_image = cv.hconcat([court_image, fitness_display_color])

        cv.imshow(WINDOWS["LIVE"], result_image)

        key = cv.waitKey(1)
        if key == ord('q'):
            running = False
        


if __name__ == "__main__":
    create_windows()

    data_provider: DataProvider = ExperimentData()
    experiments: List[ImageConfiguration] = data_provider.define_experiments()


    for experiment in experiments:
       evolve(experiment)

    
    destroy_windows()   