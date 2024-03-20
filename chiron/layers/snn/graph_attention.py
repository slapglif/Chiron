import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

class GraphAttentionLayer(nn.Module):
   def __init__(self, in_features: int, out_features: int):
       """
       Initialize the GraphAttentionLayer.

       Args:
           in_features (int): Number of input features.
           out_features (int): Number of output features.
       """
       super(GraphAttentionLayer, self).__init__()
       self.is_fitted = False  # Flag to indicate if the mask predictor has been fitted
       self.W = nn.Linear(in_features, out_features)
       self.leakyrelu = nn.LeakyReLU(0.2)
       self.mask_predictor = SGDRegressor(max_iter=1000, tol=1e-3)
       self.tfidf_vectorizer = TfidfVectorizer()

   def forward(self, x: torch.Tensor, adj_mat: torch.Tensor, conversation_texts: List[str]) -> torch.Tensor:
       """
       Perform forward pass of the GraphAttentionLayer.

       Args:
           x (torch.Tensor): Input features.
           adj_mat (torch.Tensor): Adjacency matrix.
           conversation_texts (List[str]): List of conversation texts.

       Returns:
           torch.Tensor: Output features after applying graph attention.
       """
       Wh = self.W(x)
       e = self.leakyrelu(Wh)

       data_features = self.extract_features(conversation_texts)

       if not hasattr(self, 'is_fitted') or not self.is_fitted:
           # Convert sparse tensor to dense tensor before flattening
           adj_mat_dense = adj_mat.to_dense()
           adj_mat_flat = adj_mat_dense.flatten().cpu().numpy()

           # Batch processing
           batch_size = 1024  # Adjust the batch size as per your memory constraints
           num_batches = (adj_mat_flat.shape[0] + batch_size - 1) // batch_size

           for i in range(num_batches):
               start = i * batch_size
               end = min((i + 1) * batch_size, adj_mat_flat.shape[0])
               adj_mat_batch = adj_mat_flat[start:end]

               # Repeat data_features to match the size of adj_mat_batch
               num_repeats = adj_mat_batch.shape[0] // data_features.shape[0]
               remainder = adj_mat_batch.shape[0] % data_features.shape[0]
               data_features_repeated = np.repeat(data_features, num_repeats, axis=0)
               data_features_remainder = data_features[:remainder]
               data_features_batch = np.concatenate((data_features_repeated, data_features_remainder), axis=0)

               self.mask_predictor.partial_fit(data_features_batch, adj_mat_batch)

           self.is_fitted = True

       mask_value = self.mask_predictor.predict(data_features).reshape(-1, 1)
       mask_value_tensor = torch.from_numpy(mask_value).to(e.device)
       zero_vec = mask_value_tensor.unsqueeze(-1) * torch.ones_like(e)
       attention = torch.where(adj_mat.to_dense().unsqueeze(1).expand_as(e) > 0, e, zero_vec)
       attention = F.softmax(attention, dim=-1)
       h_prime = torch.einsum('bijd,bjd->bid', attention, Wh)

       return F.elu(h_prime)

   def extract_features(self, conversation_texts: List[str]) -> np.ndarray:
       """
       Extract features from conversation texts using TF-IDF.

       Args:
           conversation_texts (List[str]): List of conversation texts.

       Returns:
           np.ndarray: Extracted features.
       """
       tfidf_features = self.tfidf_vectorizer.fit_transform(conversation_texts).toarray()
       return tfidf_features