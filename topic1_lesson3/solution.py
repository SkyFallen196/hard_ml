import math
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler


DEFAULT_N_EPOCHS = 5
DEFAULT_HIDDEN_DIM = 30
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NDCG_TOP_K = 10
DEFAULT_SEED = 0
TARGET_COLUMN = 0
QUERY_ID_COLUMN = 1


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, hidden_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        return self.model(input_1)


class Solution:
    def __init__(self, n_epochs: int = DEFAULT_N_EPOCHS, 
                 listnet_hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 lr: float = DEFAULT_LEARNING_RATE, 
                 ndcg_top_k: int = DEFAULT_NDCG_TOP_K):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _get_data(self) -> Tuple[np.ndarray, ...]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([TARGET_COLUMN, QUERY_ID_COLUMN], axis=1).values
        y_train = train_df[TARGET_COLUMN].values
        query_ids_train = train_df[QUERY_ID_COLUMN].values.astype(int)

        X_test = test_df.drop([TARGET_COLUMN, QUERY_ID_COLUMN], axis=1).values
        y_test = test_df[TARGET_COLUMN].values
        query_ids_test = test_df[QUERY_ID_COLUMN].values.astype(int)

        return X_train, y_train, query_ids_train, X_test, y_test, query_ids_test

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
         X_test, y_test, self.query_ids_test) = self._get_data()
        
        X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        X_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)
        
        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)
        self.ys_train = torch.FloatTensor(y_train)
        self.ys_test = torch.FloatTensor(y_test)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                      inp_query_ids: np.ndarray) -> np.ndarray:
        unique_query_ids = np.unique(inp_query_ids)
        for qid in unique_query_ids:
            mask = (inp_query_ids == qid)
            inp_feat_array[mask] = StandardScaler().fit_transform(inp_feat_array[mask])
        return inp_feat_array

    def _create_model(self, listnet_num_input_features: int,
                     listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(DEFAULT_SEED)
        return ListNet(listnet_num_input_features, listnet_hidden_dim)

    def fit(self) -> List[float]:
        ndcgs = []
        for _ in range(self.n_epochs):
            self._train_one_epoch()
            ndcgs.append(self._eval_test_set())
        return ndcgs

    def _calc_loss(self, batch_ys: torch.FloatTensor,
                  batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        return -(batch_ys * torch.log(batch_pred)).sum()

    def _train_one_epoch(self) -> None:
        self.model.train()
        unique_query_ids = np.unique(self.query_ids_train)
        
        for qid in unique_query_ids:
            mask = (self.query_ids_train == qid)
            batch_pred = self.model(self.X_train[mask]).flatten()
            batch_pred = torch.nn.functional.softmax(batch_pred)
            batch_true = self.ys_train[mask].flatten()
            
            loss = self._calc_loss(batch_true, batch_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            unique_query_ids = np.unique(self.query_ids_test)
            
            for qid in unique_query_ids:
                mask = (self.query_ids_test == qid)
                batch_pred = self.model(self.X_test[mask]).flatten()
                batch_true = self.ys_test[mask].flatten()
                ndcg = self._ndcg_k(batch_true, batch_pred, self.ndcg_top_k)
                ndcgs.append(ndcg)
                
            return float(np.mean(ndcgs))

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        ideal_dcg = self._dcg_k(ys_true, ys_true, ndcg_top_k)
        dcg = self._dcg_k(ys_true, ys_pred, ndcg_top_k)
        return 0.0 if ideal_dcg == 0 else dcg / ideal_dcg

    def _dcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
               top_k: int) -> float:
        _, indices = torch.sort(ys_pred, descending=True)
        ind = min(len(ys_true), top_k)
        ys_true_sorted = ys_true[indices][:ind]
        

        ranks = torch.arange(1, ind + 1, device=ys_true.device)
        gains = (2 ** ys_true_sorted - 1) / torch.log2(ranks + 1)
        return float(gains.sum())