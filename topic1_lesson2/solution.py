from math import log2
from typing import Literal, Optional
from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    """
    Вычисляет количество пар элементов, которые нужно поменять местами для получения идеального ранжирования.
    
    Args:
        ys_true: Тензор с истинными значениями релевантности
        ys_pred: Тензор с предсказанными значениями релевантности
        
    Returns:
        Количество пар элементов, требующих перестановки
    """
    _, sorted_indices = sort(ys_pred, descending=True)
    sorted_true_relevance = ys_true[sorted_indices]
    return sum(
        int(relevance_j > relevance_i)
        for i, relevance_i in enumerate(sorted_true_relevance[:-1])
        for relevance_j in sorted_true_relevance[i:]
    )


def compute_gain(y_value: float, gain_scheme: Literal['const', 'exp2']) -> float:
    """
    Вычисляет gain для заданного значения в соответствии с выбранной схемой.
    
    Args:
        y_value: Значение релевантности для вычисления gain
        gain_scheme: Тип вычисления gain ('const' или 'exp2')
        
    Returns:
        Вычисленное значение gain
    """
    if gain_scheme == 'const':
        return y_value
    elif gain_scheme == 'exp2':
        return 2 ** y_value - 1
    else:
        raise ValueError(f"Неподдерживаемый тип gain: {gain_scheme}")


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: Literal['const', 'exp2']) -> float:
    """
    Вычисляет Discounted Cumulative Gain.
    
    Args:
        ys_true: Тензор с истинными значениями релевантности
        ys_pred: Тензор с предсказанными значениями релевантности
        gain_scheme: Тип вычисления gain
        
    Returns:
        Значение DCG
    """
    _, sorted_indices = sort(ys_pred, descending=True)
    sorted_true_relevance = ys_true[sorted_indices]
    return sum(
        compute_gain(relevance.item(), gain_scheme) / log2(position + 1)
        for position, relevance in enumerate(sorted_true_relevance, start=1)
    )


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: Literal['const', 'exp2'] = 'const') -> float:
    """
    Вычисляет Normalized Discounted Cumulative Gain.
    
    Args:
        ys_true: Тензор с истинными значениями релевантности
        ys_pred: Тензор с предсказанными значениями релевантности
        gain_scheme: Тип вычисления gain
        
    Returns:
        Значение NDCG
    """
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme)
    if ideal_dcg == 0:
        return 0.0
    return dcg(ys_true, ys_pred, gain_scheme) / ideal_dcg


def precision_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> Optional[float]:
    """
    Вычисляет precision@k.
    
    Args:
        ys_true: Тензор с истинными значениями релевантности
        ys_pred: Тензор с предсказанными значениями релевантности
        k: Количество элементов для рассмотрения
        
    Returns:
        Значение precision@k или None, если невозможно вычислить
    """
    if ys_pred.sum() == 0:
        return None

    _, sorted_indices = sort(ys_pred, descending=True)
    sorted_true_relevance = ys_true[sorted_indices]
    top_k = min(len(ys_true), k)
    true_positives = sorted_true_relevance[:top_k].sum().item()
    total_positives = min(ys_true.sum().item(), k)

    return true_positives / total_positives if total_positives > 0 else 0.0


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    """
    Вычисляет Reciprocal Rank.
    
    Args:
        ys_true: Тензор с истинными значениями релевантности
        ys_pred: Тензор с предсказанными значениями релевантности
        
    Returns:
        Значение Reciprocal Rank
    """
    _, sorted_indices = sort(ys_pred, descending=True)
    sorted_true_relevance = ys_true[sorted_indices]
    first_relevant_rank = 1 + (sorted_true_relevance == 1).nonzero(as_tuple=True)[0].item()
    return 1 / first_relevant_rank


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15) -> float:
    """
    Вычисляет P-found метрику.
    
    Args:
        ys_true: Тензор с истинными значениями релевантности
        ys_pred: Тензор с предсказанными значениями релевантности
        p_break: Вероятность прерывания просмотра
        
    Returns:
        Значение P-found метрики
    """
    _, sorted_indices = sort(ys_pred, descending=True)
    sorted_true_relevance = ys_true[sorted_indices]
    
    if len(sorted_true_relevance) == 0:
        return 0.0
        
    p_found = sorted_true_relevance[0].item()
    look_probability = 1
    previous_relevance = sorted_true_relevance[0].item()
    
    if len(sorted_true_relevance) == 1:
        return p_found

    for relevance in sorted_true_relevance[1:]:
        look_probability = look_probability * (1 - previous_relevance) * (1 - p_break)
        p_found += look_probability * relevance.item()
        previous_relevance = relevance.item()

    return p_found


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> Optional[float]:
    """
    Вычисляет Average Precision.
    
    Args:
        ys_true: Тензор с истинными значениями релевантности
        ys_pred: Тензор с предсказанными значениями релевантности
        
    Returns:
        Значение Average Precision или None, если невозможно вычислить
    """
    if ys_true.sum() == 0:
        return None
        
    _, sorted_indices = sort(ys_pred, descending=True)
    sorted_true_relevance = ys_true[sorted_indices]
    
    total_precision = sum(
        sorted_true_relevance[:position].sum().item() / position
        for position, relevance in enumerate(sorted_true_relevance, start=1)
        if relevance > 0
    )
    
    return total_precision / ys_true.sum().item() 