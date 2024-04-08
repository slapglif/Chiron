import numpy as np
import torch
import pytest

from chiron.layers.htm.model import HTMSpatialPooler, HTMModel

# Test data
batch_size = 2
seq_len = 3
input_size = 10
num_minicolumns = 100
potential_radius = 10
potential_pct = 0.5
input_tensor = torch.randn(batch_size, seq_len, input_size)
connections = np.random.rand(num_minicolumns, input_size)

# Create an instance of HTMSpatialPooler
spatial_pooler = HTMSpatialPooler(
    input_size=input_size,
    minicolumn_size=16,
    potential_radius=potential_radius,
    potential_pct=potential_pct,
    global_inhibition=True,
    local_area_density=-1.0,
    num_active_columns_per_inhibition_area=10,
    stimulus_threshold=0.0,
    syn_perm_inactive_dec=0.01,
    syn_perm_active_inc=0.1,
    syn_perm_connected=0.2,
    min_pct_overlap_duty_cycle=0.005,
    duty_cycle_period=500,
    max_boost=5.0,
    seed=42,
    seq_len=seq_len,  # Add this line
)
# Create an instance of HTMModel
htm_model = HTMModel(
    sdr_dimensions=num_minicolumns,
    device=torch.device("cpu"),
    input_size=input_size,
    minicolumn_size=16,
    potential_radius=potential_radius,
    potential_pct=potential_pct,
    global_inhibition=True,
    local_area_density=-1.0,
    num_active_columns_per_inhibition_area=10,
    stimulus_threshold=0.0,
    syn_perm_inactive_dec=0.01,
    syn_perm_active_inc=0.1,
    syn_perm_connected=0.2,
    min_pct_overlap_duty_cycle=0.005,
    duty_cycle_period=500,
    max_boost=5.0,
    seed=42,
    seq_len=seq_len,  # Add this line
)


# Test HTMSpatialPooler.compute_overlap
def test_htm_spatial_pooler_compute_overlap():
    overlap = spatial_pooler.compute_overlap(input_tensor.numpy())
    assert overlap.shape == (batch_size, seq_len, num_minicolumns)
    assert np.all(overlap >= 0)  # Ensure overlap scores are non-negative


# Test HTMSpatialPooler.forward
def test_htm_spatial_pooler_forward():
    active_columns = spatial_pooler.forward(input_tensor.numpy())
    assert active_columns.shape == (batch_size, seq_len, num_minicolumns)
    assert np.all(
        (active_columns >= 0) & (active_columns <= 1)
    )  # Ensure active columns are binary


# Test HTMModel.forward
def test_htm_model_forward():
    output = htm_model(input_tensor)
    assert output.shape == (batch_size, seq_len, num_minicolumns)
    assert torch.all((output >= 0) & (output <= 1))  # Ensure output is binary


# Test HTMModel.inspect
def test_htm_model_inspect():
    inspection_data = htm_model.inspect()
    assert isinstance(inspection_data, dict)
    assert "connections" in inspection_data
    assert "permanences" in inspection_data
    assert inspection_data["connections"].shape == (num_minicolumns, input_size)
    assert inspection_data["permanences"].shape == (num_minicolumns, input_size)


# Test HTMSpatialPooler.compute with custom input
@pytest.mark.parametrize(
    "input_vector", [np.random.rand(batch_size * seq_len, input_size)]
)
def test_htm_spatial_pooler_compute(input_vector):
    overlap = spatial_pooler.compute_overlap(input_vector)
    active_columns = spatial_pooler.compute(overlap)
    assert active_columns.shape == (batch_size, seq_len, num_minicolumns)
    assert np.all(
        (active_columns >= 0) & (active_columns <= 1)
    )  # Ensure active columns are binary


# Test HTMSpatialPooler.compute_overlap with different input shapes
@pytest.mark.parametrize(
    "input_shape",
    [(batch_size, seq_len, input_size), (batch_size * seq_len, input_size)],
)
def test_htm_spatial_pooler_compute_overlap_shapes(input_shape):
    input_vector = np.random.rand(*input_shape)
    overlap = spatial_pooler.compute_overlap(input_vector)
    assert overlap.shape == (batch_size, seq_len, num_minicolumns)
