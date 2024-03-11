import numpy as np
from typing import List


class SDRGenerator:
    def __init__(self, num_active_bits: int = 20, sdr_size: int = 1000):
        self.num_active_bits = num_active_bits
        self.sdr_size = sdr_size

    def generate_sdr(self, embedding: List[float]) -> np.ndarray:
        indices = np.argsort(embedding)[-self.num_active_bits :]
        sdr = np.zeros(self.sdr_size, dtype=np.uint8)
        sdr[indices] = 1
        return sdr

    def generate_sdrs(self, embeddings: List[List[float]]) -> np.ndarray:
        sdrs = []
        for embedding in embeddings:
            sdr = self.generate_sdr(embedding)
            sdrs.append(sdr)
        return np.array(sdrs, dtype=np.uint8)
