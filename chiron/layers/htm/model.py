# chiron/layers/htm/model.py

from typing import Union, Tuple

import numpy as np
import torch
from loguru import logger
from torch import nn


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
        self.input_size = input_size
        self.minicolumn_size = minicolumn_size
        self.num_minicolumns = (input_size + minicolumn_size - 1) // minicolumn_size
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

        # Initialize the connections and permanence matrices
        self.connections = np.random.rand(self.num_minicolumns, self.input_size)
        self.connections[self.connections < self.syn_perm_connected] = 0
        self.connections[self.connections >= self.syn_perm_connected] = 1

        self.permanences = np.zeros((self.num_minicolumns, self.input_size))

        # Initialize the duty cycles and boosting factors
        self.active_duty_cycles = np.zeros(self.num_minicolumns)
        self.overlap_duty_cycles = np.zeros(self.num_minicolumns)
        self.min_overlap_duty_cycles = np.zeros(self.num_minicolumns)
        self.boosting_factors = np.ones(self.num_minicolumns)

    def compute_overlap(
        self, input_vector: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute the overlap between the input vector and the minicolumn connections using einsum.

        Args:
            input_vector (Union[np.ndarray, torch.Tensor]): The input vector of shape (batch_size, seq_len, input_size) or (batch_size * seq_len, input_size).

        Returns:
            Union[np.ndarray, torch.Tensor]: The overlap scores for each minicolumn of shape (batch_size, seq_len, num_minicolumns) or (batch_size * seq_len, num_minicolumns).
        """
        if isinstance(input_vector, torch.Tensor):
            input_vector = input_vector.cpu().numpy()

        input_shape = input_vector.shape
        if len(input_shape) == 3:
            batch_size, seq_len, input_size = input_shape
            input_vector = input_vector.reshape(batch_size * seq_len, input_size)
        elif len(input_shape) == 2:
            batch_size_seq_len, input_size = input_shape
        else:
            raise ValueError(f"Invalid input shape: {input_shape}")

        # Ensure the connections matrix has the correct shape
        assert self.connections.shape == (
            self.num_minicolumns,
            input_size,
        ), f"Connections matrix shape {self.connections.shape} does not match expected shape ({self.num_minicolumns}, {input_size})"

        overlap = np.einsum("ij,kj->ik", input_vector, self.connections)

        if len(input_shape) == 3:
            overlap = overlap.reshape(batch_size, seq_len, self.num_minicolumns)
        else:
            overlap = overlap.reshape(batch_size_seq_len, self.num_minicolumns)

        return overlap

    def inhibit_columns(self, overlap: np.ndarray) -> np.ndarray:
        """
        Perform inhibition on the minicolumns based on the overlap scores.

        Args:
            overlap (np.ndarray): The overlap scores for each minicolumn of shape (batch_size, seq_len, num_minicolumns) or (batch_size * seq_len, num_minicolumns).

        Returns:
            np.ndarray: The active columns after inhibition of shape (batch_size, seq_len, num_minicolumns) or (batch_size * seq_len, num_minicolumns).
        """
        input_shape = overlap.shape
        if len(input_shape) == 3:
            batch_size, seq_len, _ = input_shape
            active_columns = np.zeros(
                (batch_size, seq_len, self.num_minicolumns), dtype=bool
            )
        else:
            batch_size_seq_len, _ = input_shape
            active_columns = np.zeros(
                (batch_size_seq_len, self.num_minicolumns), dtype=bool
            )

        if self.global_inhibition:
            num_active = np.minimum(
                self.num_active_columns_per_inhibition_area, self.num_minicolumns
            )
            thresholds = np.partition(overlap, -num_active, axis=-1)[:, -num_active]
            active_columns = overlap >= thresholds[:, np.newaxis]
        else:
            raise NotImplementedError("Local inhibition is not implemented.")

        return active_columns

    def update_permanences(
        self, active_columns: np.ndarray, input_vector: np.ndarray
    ) -> None:
        """
        Update the permanences of the synapses based on the active columns and input vector.

        Args:
            active_columns (np.ndarray): The active columns of shape (batch_size, seq_len, num_minicolumns) or (batch_size * seq_len, num_minicolumns).
            input_vector (np.ndarray): The input vector of shape (batch_size, seq_len, input_size) or (batch_size * seq_len, input_size).
        """
        input_shape = input_vector.shape
        if len(input_shape) == 3:
            batch_size, seq_len, input_size = input_shape
        elif len(input_shape) == 2:
            batch_size_seq_len, input_size = input_shape
        else:
            raise ValueError(f"Invalid input shape: {input_shape}")

        # Update the permanences for active columns
        active_columns_indices = np.transpose(np.nonzero(active_columns))
        for batch_seq_idx, col_idx in active_columns_indices:
            if len(input_shape) == 3:
                batch_idx, seq_idx = batch_seq_idx // seq_len, batch_seq_idx % seq_len
                self.permanences[col_idx] += (
                    self.syn_perm_active_inc * input_vector[batch_idx, seq_idx]
                )
            else:
                self.permanences[col_idx] += (
                    self.syn_perm_active_inc * input_vector[batch_seq_idx]
                )

        # Decay the permanences for inactive columns
        inactive_columns = ~active_columns
        for batch_seq_idx in range(inactive_columns.shape[0]):
            for col_idx in range(self.num_minicolumns):
                if (
                    inactive_columns[batch_seq_idx, col_idx]
                    and self.connections[col_idx].any()
                ):
                    if len(input_shape) == 3:
                        batch_idx, seq_idx = (
                            batch_seq_idx // seq_len,
                            batch_seq_idx % seq_len,
                        )
                        self.permanences[col_idx] -= self.syn_perm_inactive_dec
                    else:
                        self.permanences[col_idx] -= self.syn_perm_inactive_dec

        # Clip the permanences to the range [0, 1]
        self.permanences = np.clip(self.permanences, 0, 1)

        # Update the connections based on the permanences
        self.connections = np.where(
            self.permanences >= self.syn_perm_connected, 1.0, 0.0
        )

    def update_duty_cycles(
        self, overlap: np.ndarray, active_columns: np.ndarray
    ) -> None:
        """
        Update the duty cycles of the minicolumns.

        Args:
            overlap (np.ndarray): The overlap scores for each minicolumn of shape (batch_size, seq_len, num_minicolumns) or (batch_size * seq_len, num_minicolumns).
            active_columns (np.ndarray): The active columns of shape (batch_size, seq_len, num_minicolumns) or (batch_size * seq_len, num_minicolumns).
        """
        input_shape = overlap.shape
        if len(input_shape) == 3:
            batch_size, seq_len, _ = input_shape
            overlap = overlap.reshape(batch_size * seq_len, self.num_minicolumns)
            active_columns = active_columns.reshape(
                batch_size * seq_len, self.num_minicolumns
            )
        else:
            batch_size_seq_len, _ = input_shape

        # Update the active duty cycles
        self.active_duty_cycles = (
            self.active_duty_cycles * (self.duty_cycle_period - 1)
            + np.sum(active_columns, axis=0)
        ) / self.duty_cycle_period

        # Update the overlap duty cycles
        self.overlap_duty_cycles = (
            self.overlap_duty_cycles * (self.duty_cycle_period - 1)
            + np.sum(overlap, axis=0)
        ) / self.duty_cycle_period

        # Update the minimum overlap duty cycles
        self.min_overlap_duty_cycles = (
            self.active_duty_cycles * self.min_pct_overlap_duty_cycle
        )

    def update_boosting_factors(self) -> None:
        """
        Update the boosting factors of the minicolumns.
        """
        self.boosting_factors = np.exp(
            self.max_boost * (self.min_overlap_duty_cycles - self.overlap_duty_cycles)
        ).clip(1, self.max_boost)

    def compute(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Compute the active columns for a given input vector.

        Args:
            input_vector (np.ndarray): The input vector of shape (batch_size, seq_len, input_size).

        Returns:
            np.ndarray: The active columns of shape (batch_size, seq_len, num_minicolumns) or (batch_size * seq_len, num_minicolumns).
        """
        batch_size, seq_len, input_size = input_vector.shape
        input_vector_reshaped = input_vector.reshape(batch_size * seq_len, input_size)

        overlap = self.compute_overlap(input_vector_reshaped)
        active_columns = self.inhibit_columns(overlap * self.boosting_factors)
        self.update_permanences(active_columns, input_vector_reshaped)
        self.update_duty_cycles(overlap, active_columns)
        self.update_boosting_factors()

        if len(input_vector_reshaped.shape) == 2:
            active_columns = active_columns.reshape(
                batch_size, seq_len, self.num_minicolumns
            )
        else:
            active_columns = active_columns.reshape(
                batch_size * seq_len, self.num_minicolumns
            )

        return active_columns

    def forward(
        self, input_vector: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Perform the forward pass of the spatial pooler.

        Args:
            input_vector (Union[np.ndarray, torch.Tensor]): The input vector of shape (batch_size, seq_len, input_size).

        Returns:
            Union[np.ndarray, torch.Tensor]: The active columns of shape (batch_size, seq_len, num_minicolumns) or (batch_size * seq_len, num_minicolumns).
        """
        if isinstance(input_vector, torch.Tensor):
            input_vector = input_vector.cpu().numpy()

        active_columns = self.compute(input_vector)

        if isinstance(input_vector, torch.Tensor):
            if active_columns.ndim == 3:
                active_columns = (
                    torch.from_numpy(active_columns).float().to(input_vector.device)
                )
            else:
                active_columns = (
                    torch.from_numpy(active_columns)
                    .float()
                    .to(input_vector.device)
                    .unsqueeze(1)
                )

        return active_columns


class HTMModel(nn.Module):
    def __init__(self, sdr_dimensions: int, device: torch.device, **kwargs):
        """
        Initialize the HTMModel.

        Args:
            sdr_dimensions (int): The dimensions of the Sparse Distributed Representation (SDR).
            device (torch.device): The device to use for the model (e.g., torch.device('cuda') or torch.device('cpu')).
            **kwargs: Additional arguments for the spatial pooler.
        """
        super(HTMModel, self).__init__()
        self.spatial_pooler = HTMSpatialPooler(**kwargs)
        self.output_size = sdr_dimensions
        self.device = device
        self.fc = nn.Linear(self.spatial_pooler.num_minicolumns, sdr_dimensions).to(
            device
        )

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Process the input sequence using the HTM model.

        Args:
            input_sequence (torch.Tensor): The sequence of input vectors of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: The sequence of active columns of shape (batch_size, seq_len, num_minicolumns) or (batch_size * seq_len, num_minicolumns).
        """
        batch_size, seq_len, input_size = input_sequence.shape

        # Resize the input sequence to match the expected input size
        expected_input_size = self.spatial_pooler.input_size
        if input_size != expected_input_size:
            logger.warning(
                f"Input sequence feature size {input_size} does not match the expected input size {expected_input_size}. Resizing the input sequence."
            )
            input_sequence = input_sequence[:, :, :expected_input_size]

        # Compute the active columns for the input sequence
        active_columns = self.spatial_pooler.forward(input_sequence)

        # Convert active columns to a PyTorch tensor
        if active_columns.ndim == 3:
            active_columns_tensor = torch.tensor(
                active_columns, dtype=torch.float32, device=self.device
            )
        else:
            active_columns_tensor = torch.tensor(
                active_columns, dtype=torch.float32, device=self.device
            ).unsqueeze(1)

        logger.debug(
            f"Performing HTM forward pass on input sequence of shape {input_sequence.shape}"
        )
        logger.debug(self.inspect())

        # Apply the fully connected layer
        output = self.fc(active_columns_tensor)

        if active_columns.ndim == 2:
            output = output.squeeze(1)

        return output

    def inspect(self) -> dict:
        """
        View the HTM model parameters.

        Returns:
            dict: A dictionary containing the HTM model parameters.
        """
        inspection_data = {
            "connections": self.spatial_pooler.connections,
            "permanences": self.spatial_pooler.permanences,
            "boosting_factors": self.spatial_pooler.boosting_factors,
            "active_duty_cycles": self.spatial_pooler.active_duty_cycles,
            "overlap_duty_cycles": self.spatial_pooler.overlap_duty_cycles,
            "min_overlap_duty_cycles": self.spatial_pooler.min_overlap_duty_cycles,
            "min_pct_overlap_duty_cycle": self.spatial_pooler.min_pct_overlap_duty_cycle,
            "duty_cycle_period": self.spatial_pooler.duty_cycle_period,
            "max_boost": self.spatial_pooler.max_boost,
            "global_inhibition": self.spatial_pooler.global_inhibition,
            "num_active_columns_per_inhibition_area": self.spatial_pooler.num_active_columns_per_inhibition_area,
        }
        return inspection_data
