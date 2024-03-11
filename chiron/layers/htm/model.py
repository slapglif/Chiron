# htm/model.py

from typing import List

import numpy as np
import torch
from torch import nn

from chiron.utils.config import Config


class HTMSpatialPooler:
    def __init__(
        self,
        input_size: int,
        minicolumn_size: int,
        potential_radius: int,
        potential_pct: float,
        global_inhibition: bool,
        local_area_density: float,
        num_active_columns_per_inhibition_area: int,
        stimulus_threshold: float,
        syn_perm_inactive_dec: float,
        syn_perm_active_inc: float,
        syn_perm_connected: float,
        min_pct_overlap_duty_cycle: float,
        duty_cycle_period: int,
        max_boost: float,
        seed: int,

    ):
        """
        Initialize the HTMSpatialPooler.

        Args:
            input_size (int): The size of the input vector.
            minicolumn_size (int): The size of a minicolumn.
            potential_radius (int): The radius of potential connections.
            potential_pct (float): The percentage of potential connections.
            global_inhibition (bool): Whether to use global inhibition.
            local_area_density (float): The density of active columns within a local inhibition area.
            num_active_columns_per_inhibition_area (int): The number of active columns per inhibition area.
            stimulus_threshold (float): The stimulus threshold for a synapse to be considered active.
            syn_perm_inactive_dec (float): The amount by which an inactive synapse's permanence value is decremented.
            syn_perm_active_inc (float): The amount by which an active synapse's permanence value is incremented.
            syn_perm_connected (float): The permanence value above which a synapse is considered connected.
            min_pct_overlap_duty_cycle (float): The minimum percentage of overlap duty cycle before inhibition.
            duty_cycle_period (int): The period used to calculate duty cycles.
            max_boost (float): The maximum boost value.
            seed (int): The random seed.
        """
        self.inhibition_radius = None
        self.min_overlap_duty_cycles = None
        self.overlap_duty_cycles = None
        self.permanences = None
        self.active_duty_cycles = None
        self.boosting_factors = None
        self.connections = None
        self.input_size = input_size
        self.minicolumn_size = minicolumn_size
        self.potential_radius = potential_radius
        self.potential_pct = potential_pct
        self.global_inhibition = global_inhibition
        self.local_area_density = local_area_density
        self.num_active_columns_per_inhibition_area = (
            num_active_columns_per_inhibition_area
        )
        self.stimulus_threshold = stimulus_threshold
        self.syn_perm_inactive_dec = syn_perm_inactive_dec
        self.syn_perm_active_inc = syn_perm_active_inc
        self.syn_perm_connected = syn_perm_connected
        self.min_pct_overlap_duty_cycle = min_pct_overlap_duty_cycle
        self.duty_cycle_period = duty_cycle_period
        self.max_boost = max_boost
        self.seed = seed

        self.num_minicolumns = self.input_size // self.minicolumn_size
        self.minicolumn_dimensions = (self.num_minicolumns,)
        self.input_dimensions = (self.input_size,)

        self.initialize_connections()
        self.initialize_boosting()

    def initialize_connections(self) -> None:
        """
        Initialize the connections of the spatial pooler.
        """
        np.random.seed(self.seed)
        self.connections = np.random.rand(self.num_minicolumns, self.input_size)
        self.connections[self.connections < self.syn_perm_connected] = 0
        self.connections[self.connections >= self.syn_perm_connected] = 1

        self.permanences = np.zeros((self.num_minicolumns, self.input_size))
        for i in range(self.num_minicolumns):
            potential_indices = self.choose_potential_connections(i)
            self.permanences[i, potential_indices] = np.random.rand(
                len(potential_indices)
            )

    def initialize_boosting(self) -> None:
        """
        Initialize the boosting parameters of the spatial pooler.
        """
        self.boosting_factors = np.ones(self.num_minicolumns)
        self.active_duty_cycles = np.zeros(self.num_minicolumns)
        self.overlap_duty_cycles = np.zeros(self.num_minicolumns)
        self.min_overlap_duty_cycles = np.zeros(self.num_minicolumns)

    def choose_potential_connections(self, minicolumn_index: int) -> np.ndarray:
        """
        Choose the potential connections for a given minicolumn.

        Args:
            minicolumn_index (int): The index of the minicolumn.

        Returns:
            np.ndarray: The indices of the potential connections.
        """
        center = minicolumn_index * self.minicolumn_size + self.minicolumn_size // 2
        radius = self.potential_radius
        start = max(0, center - radius)
        end = min(self.input_size, center + radius + 1)
        indices = np.arange(start, end)
        np.random.shuffle(indices)
        num_potential = int(len(indices) * self.potential_pct)
        return indices[:num_potential]

    def compute_overlap(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Compute the overlap between the input vector and the minicolumn connections.

        Args:
            input_vector (np.ndarray): The input vector.

        Returns:
            np.ndarray: The overlap scores for each minicolumn.
        """
        overlap = np.zeros(self.num_minicolumns)
        for i in range(self.num_minicolumns):
            connected_synapses = self.connections[i]
            overlap[i] = np.sum(input_vector * connected_synapses)
        return overlap

    def inhibit_columns(self, overlap: np.ndarray) -> np.ndarray:
        """
        Perform inhibition on the minicolumns based on the overlap scores.

        Args:
            overlap (np.ndarray): The overlap scores for each minicolumn.

        Returns:
            np.ndarray: The active columns after inhibition.
        """
        if self.global_inhibition:
            return self.global_inhibition_func(overlap)
        else:
            return self.local_inhibition_func(overlap)

    def global_inhibition_func(self, overlap: np.ndarray) -> np.ndarray:
        """
        Perform global inhibition on the minicolumns.

        Args:
            overlap (np.ndarray): The overlap scores for each minicolumn.

        Returns:
            np.ndarray: The active columns after global inhibition.
        """
        winners = np.argsort(overlap)[-self.num_active_columns_per_inhibition_area :]
        active_columns = np.zeros(self.num_minicolumns)
        active_columns[winners] = 1
        return active_columns

    def local_inhibition_func(self, overlap: np.ndarray) -> np.ndarray:
        """
        Perform local inhibition on the minicolumns.

        Args:
            overlap (np.ndarray): The overlap scores for each minicolumn.

        Returns:
            np.ndarray: The active columns after local inhibition.
        """
        active_columns = np.zeros(self.num_minicolumns)
        for i in range(self.num_minicolumns):
            neighborhood = self.get_neighborhood(i)
            neighborhood_overlap = overlap[neighborhood]
            max_overlap = np.max(neighborhood_overlap)
            if overlap[i] >= max_overlap:
                active_columns[i] = 1
        return active_columns

    def get_neighborhood(self, minicolumn_index: int) -> np.ndarray:
        """
        Get the neighborhood of a given minicolumn.

        Args:
            minicolumn_index (int): The index of the minicolumn.

        Returns:
            np.ndarray: The indices of the minicolumns in the neighborhood.
        """
        start = max(0, minicolumn_index - self.inhibition_radius)
        end = min(self.num_minicolumns, minicolumn_index + self.inhibition_radius + 1)
        return np.arange(start, end)

    def update_permanences(
        self, active_columns: np.ndarray, input_vector: np.ndarray
    ) -> None:
        """
        Update the permanences of the synapses based on the active columns and input vector.

        Args:
            active_columns (np.ndarray): The active columns.
            input_vector (np.ndarray): The input vector.
        """
        for i in range(self.num_minicolumns):
            if active_columns[i]:
                self.permanences[i, input_vector == 1] += self.syn_perm_active_inc
                self.permanences[i, input_vector == 0] -= self.syn_perm_inactive_dec
            else:
                self.permanences[i, input_vector == 1] -= self.syn_perm_inactive_dec
        self.permanences[self.permanences < 0] = 0
        self.permanences[self.permanences > 1] = 1
        self.connections = np.where(self.permanences >= self.syn_perm_connected, 1, 0)

    def update_duty_cycles(
        self, overlap: np.ndarray, active_columns: np.ndarray
    ) -> None:
        """
        Update the duty cycles of the minicolumns.

        Args:
            overlap (np.ndarray): The overlap scores for each minicolumn.
            active_columns (np.ndarray): The active columns.
        """
        self.active_duty_cycles = (
            self.active_duty_cycles * (self.duty_cycle_period - 1) + active_columns
        ) / self.duty_cycle_period
        self.overlap_duty_cycles = (
            self.overlap_duty_cycles * (self.duty_cycle_period - 1) + overlap
        ) / self.duty_cycle_period
        self.min_overlap_duty_cycles = (
            self.active_duty_cycles * self.min_pct_overlap_duty_cycle
        )

    def update_boosting_factors(self) -> None:
        """
        Update the boosting factors of the minicolumns.
        """
        self.boosting_factors = np.exp(
            self.max_boost * (self.min_overlap_duty_cycles - self.overlap_duty_cycles)
        )  # noqa: E501

    def compute(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Compute the active columns for a given input vector.

        Args:
            input_vector (np.ndarray): The input vector.

        Returns:
            np.ndarray: The active columns.
        """
        overlap = self.compute_overlap(input_vector)
        active_columns = self.inhibit_columns(overlap * self.boosting_factors)
        self.update_permanences(active_columns, input_vector)
        self.update_duty_cycles(overlap, active_columns)
        self.update_boosting_factors()
        return active_columns

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        """
        Compute the active columns for a given input vector.

        Args:
            input_vector (torch.Tensor): The input vector.

        Returns:
            torch.Tensor: The active columns.
        """
        input_vector = input_vector.cpu().detach().numpy()
        active_columns = self.compute(input_vector)
        return torch.from_numpy(active_columns).float()


class HTMModel(nn.Module):
    def __init__(self, **kwargs):
        """
        Initialize the HTMModel.

        Args:
            sp_params (dict): The parameters for the spatial pooler.
        """
        self.spatial_pooler = HTMSpatialPooler(**kwargs)
        self.device = torch.device("cpu")
        super(HTMModel, self).__init__()

    def forward(self, input_sequence: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Process the input sequence using the HTM model.

        Args:
            input_sequence (List[torch.Tensor]): The sequence of input vectors.

        Returns:
            List[torch.Tensor]: The sequence of active columns.
        """
        active_columns_sequence = []
        for input_vector in input_sequence:
            active_columns = self.spatial_pooler.forward(
                input_vector
            )  # Use self.spatial_pooler.forward() instead of self.spatial_pooler()
            active_columns_sequence.append(active_columns)
        return active_columns_sequence

    def inspect(self):
        """
        View the HTM model.
        """
        print(self.spatial_pooler)
        print(self.spatial_pooler.connections)
        print(self.spatial_pooler.permanences)
        print(self.spatial_pooler.boosting_factors)
        print(self.spatial_pooler.active_duty_cycles)
        print(self.spatial_pooler.overlap_duty_cycles)
        print(self.spatial_pooler.min_overlap_duty_cycles)
        print(self.spatial_pooler.min_pct_overlap_duty_cycle)
        print(self.spatial_pooler.duty_cycle_period)
        print(self.spatial_pooler.max_boost)
        print(self.spatial_pooler.global_inhibition)
        print(self.spatial_pooler.inhibition_radius)
        print(self.spatial_pooler.num_active_columns_per_inhibition_area)
