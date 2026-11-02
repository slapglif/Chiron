//! Minimal CSR sparse matrix, mirroring the scipy.sparse.csr_matrix adjacency
//! used throughout the Python pipeline.

use ndarray::ArrayView2;

/// Compressed sparse row matrix of f32 values.
#[derive(Clone, Debug)]
pub struct CsrMatrix {
    pub rows: usize,
    pub cols: usize,
    /// Row pointers, length rows + 1.
    pub indptr: Vec<usize>,
    /// Column indices, length nnz, sorted within each row.
    pub indices: Vec<usize>,
    /// Values, length nnz.
    pub data: Vec<f32>,
}

impl CsrMatrix {
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Column indices and values of one row.
    #[inline]
    pub fn row(&self, i: usize) -> (&[usize], &[f32]) {
        let (lo, hi) = (self.indptr[i], self.indptr[i + 1]);
        (&self.indices[lo..hi], &self.data[lo..hi])
    }

    /// Build from a dense matrix, keeping entries with |v| > 0.
    pub fn from_dense(dense: &ArrayView2<f32>) -> Self {
        let (rows, cols) = dense.dim();
        let mut indptr = Vec::with_capacity(rows + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for i in 0..rows {
            for j in 0..cols {
                let v = dense[(i, j)];
                if v != 0.0 {
                    indices.push(j);
                    data.push(v);
                }
            }
            indptr.push(indices.len());
        }
        Self { rows, cols, indptr, indices, data }
    }

    /// Build a k-nearest-neighbour cosine-similarity adjacency from row
    /// embeddings, mirroring the Python adjacency construction
    /// (cosine similarity -> threshold -> sparse CSR, with top_k cap).
    ///
    /// Rows of `embeddings` are L2-normalised internally; entries with
    /// similarity > `threshold` are kept, capped at the `top_k` strongest
    /// per row.
    pub fn cosine_adjacency(embeddings: &ArrayView2<f32>, threshold: f32, top_k: usize) -> Self {
        use ndarray::Axis;
        let n = embeddings.nrows();
        // L2-normalise rows so the Gram matrix is cosine similarity.
        let mut normed = embeddings.to_owned();
        for mut row in normed.axis_iter_mut(Axis(0)) {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            row.mapv_inplace(|v| v / norm);
        }
        let sims = normed.dot(&normed.t()); // (n, n) one GEMM

        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(n);
        for i in 0..n {
            candidates.clear();
            for j in 0..n {
                let s = sims[(i, j)];
                if i != j && s > threshold {
                    candidates.push((j, s));
                }
            }
            if candidates.len() > top_k {
                candidates.select_nth_unstable_by(top_k - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.truncate(top_k);
            }
            candidates.sort_unstable_by_key(|&(j, _)| j);
            for &(j, s) in &candidates {
                indices.push(j);
                data.push(s);
            }
            indptr.push(indices.len());
        }
        Self { rows: n, cols: n, indptr, indices, data }
    }

    /// Top-left square sub-matrix view (`adj[:s, :s]`), as used when slicing
    /// the cached adjacency down to the batch sequence length.
    pub fn principal_submatrix(&self, s: usize) -> CsrMatrix {
        assert!(s <= self.rows && s <= self.cols);
        let mut indptr = Vec::with_capacity(s + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for i in 0..s {
            let (cols, vals) = self.row(i);
            for (&j, &v) in cols.iter().zip(vals) {
                if j < s {
                    indices.push(j);
                    data.push(v);
                }
            }
            indptr.push(indices.len());
        }
        CsrMatrix { rows: s, cols: s, indptr, indices, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn from_dense_roundtrip() {
        let d = array![[0.0_f32, 1.0, 0.0], [2.0, 0.0, 3.0], [0.0, 0.0, 0.0]];
        let csr = CsrMatrix::from_dense(&d.view());
        assert_eq!(csr.nnz(), 3);
        assert_eq!(csr.row(1), (&[0usize, 2][..], &[2.0_f32, 3.0][..]));
        assert_eq!(csr.row(2).0.len(), 0);
    }

    #[test]
    fn cosine_adjacency_self_excluded_and_capped() {
        let e = array![
            [1.0_f32, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ];
        let csr = CsrMatrix::cosine_adjacency(&e.view(), 0.5, 1);
        // Row 0 and row 1 are similar; row 2 orthogonal to row 0.
        let (cols0, _) = csr.row(0);
        assert_eq!(cols0, &[1]);
        for i in 0..3 {
            assert!(!csr.row(i).0.contains(&i), "self loop in row {i}");
        }
    }

    #[test]
    fn principal_submatrix_filters() {
        let d = array![[0.0_f32, 1.0, 2.0], [3.0, 0.0, 4.0], [5.0, 6.0, 0.0]];
        let csr = CsrMatrix::from_dense(&d.view());
        let sub = csr.principal_submatrix(2);
        assert_eq!(sub.rows, 2);
        assert_eq!(sub.row(0), (&[1usize][..], &[1.0_f32][..]));
        assert_eq!(sub.row(1), (&[0usize][..], &[3.0_f32][..]));
    }
}
