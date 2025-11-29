# hierarchical_bayesian.py
import struct
import os
import math
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from config import *

class HierarchicalBayesianModel:
    """Иерархическая Байесовская модель для улучшения предсказаний"""
    
    def __init__(self, name: str):
        self.name = name
        
        # Гиперпараметры для иерархической модели
        self.alpha_hyper_prior = HIERARCHICAL_ALPHA_PRIOR
        self.beta_hyper_prior = HIERARCHICAL_BETA_PRIOR
        
        # Групповые статистики (группа -> статистика)
        self.group_stats: Dict[int, Tuple[float, float]] = {}  # group -> (alpha, beta)
        
        # Отображение Crumb -> группа
        self.crumb_to_group: Dict[int, int] = {}
        
        # Контекстные модели (последовательности -> статистика)
        self.context_models: Dict[tuple, Tuple[int, int]] = {}  # context -> (successes, total)
        
        # Файлы для сохранения
        self.group_file = os.path.join(HIERARCHICAL_DIR, f"{name}_groups.bin")
        self.context_file = os.path.join(HIERARCHICAL_DIR, f"{name}_context.bin")
        
        # Статистика работы модели
        self.hierarchical_predictions = 0
        self.hierarchical_successes = 0

    def assign_to_group(self, crumb: int) -> int:
        """Назначить Crumb в группу на основе его значения"""
        # Простая группировка по значению Crumb'а
        group = crumb % GROUP_SIZE
        self.crumb_to_group[crumb] = group
        return group

    def update_group_statistics(self, crumb: int, success: bool) -> None:
        """Обновить статистику группы с использованием иерархической модели"""
        group = self.assign_to_group(crumb)
        
        if group not in self.group_stats:
            # Инициализация с априорными значениями
            self.group_stats[group] = (self.alpha_hyper_prior, self.beta_hyper_prior)
        
        alpha, beta = self.group_stats[group]
        
        # Обновление гиперпараметров на основе наблюдения
        if success:
            new_alpha = alpha + 1
            new_beta = beta
        else:
            new_alpha = alpha
            new_beta = beta + 1
        
        # Применение сглаживания
        self.group_stats[group] = (new_alpha, new_beta)

    def update_context_model(self, context: List[int], next_crumb: int, success: bool) -> None:
        """Обновить контекстную модель"""
        context_key = tuple(context[-CONTEXT_DEPTH:]) if context else tuple()
        
        if context_key not in self.context_models:
            self.context_models[context_key] = (0, 0)
        
        successes, total = self.context_models[context_key]
        if success:
            successes += 1
        total += 1
        
        self.context_models[context_key] = (successes, total)

    def calculate_hierarchical_probability(self, crumb: int, context: List[int]) -> float:
        """Вычислить вероятность с использованием иерархической модели"""
        group = self.crumb_to_group.get(crumb, crumb % GROUP_SIZE)
        
        # Базовая вероятность из групповой статистики
        if group in self.group_stats:
            alpha, beta = self.group_stats[group]
            group_prob = alpha / (alpha + beta)
        else:
            group_prob = 0.5  # Априорная вероятность по умолчанию
        
        # Корректировка на основе контекста
        context_key = tuple(context[-CONTEXT_DEPTH:]) if context else tuple()
        context_prob = self._get_context_probability(context_key)
        
        # Взвешенное объединение вероятностей
        if context_prob is not None:
            # Больший вес контексту при наличии достаточных данных
            total_weight = alpha + beta if group in self.group_stats else 1
            context_weight = min(10, total_weight / 2)  # Ограничение веса контекста
            hierarchical_prob = (group_prob * total_weight + context_prob * context_weight) / (total_weight + context_weight)
        else:
            hierarchical_prob = group_prob
        
        return hierarchical_prob

    def _get_context_probability(self, context_key: tuple) -> Optional[float]:
        """Получить вероятность на основе контекста"""
        if context_key in self.context_models:
            successes, total = self.context_models[context_key]
            if total >= MIN_OBSERVATIONS:
                return successes / total
        return None

    def hierarchical_predict(self, available_crumbs: List[int], context: List[int]) -> Optional[Tuple[int, float]]:
        """Предсказание с использованием иерархической модели"""
        if not available_crumbs:
            return None
        
        best_crumb = None
        best_probability = 0.0
        
        for crumb in available_crumbs:
            probability = self.calculate_hierarchical_probability(crumb, context)
            
            if probability > best_probability:
                best_probability = probability
                best_crumb = crumb
        
        if best_crumb is not None and best_probability > CONFIDENCE_THRESHOLD:
            self.hierarchical_predictions += 1
            return (best_crumb, best_probability)
        
        return None

    def record_hierarchical_result(self, predicted_crumb: int, actual_crumb: int, context: List[int]) -> None:
        """Записать результат иерархического предсказания"""
        success = (predicted_crumb == actual_crumb)
        if success:
            self.hierarchical_successes += 1
        
        # Обновить групповую статистику
        self.update_group_statistics(actual_crumb, success)
        
        # Обновить контекстную модель
        self.update_context_model(context, actual_crumb, success)

    def get_hierarchical_accuracy(self) -> float:
        """Получить точность иерархических предсказаний"""
        if self.hierarchical_predictions == 0:
            return 0.0
        return self.hierarchical_successes / self.hierarchical_predictions

    def save_state(self) -> None:
        """Сохранить состояние иерархической модели"""
        os.makedirs(HIERARCHICAL_DIR, exist_ok=True)
        
        # Сохранить групповую статистику
        with open(self.group_file, 'wb') as f:
            # Заголовок: количество групп
            f.write(struct.pack(">I", len(self.group_stats)))
            # Данные: group, alpha, beta
            for group, (alpha, beta) in self.group_stats.items():
                f.write(struct.pack(">Iff", group, alpha, beta))
        
        # Сохранить контекстные модели
        with open(self.context_file, 'wb') as f:
            # Заголовок: количество контекстов
            f.write(struct.pack(">I", len(self.context_models)))
            # Данные: context_length, context..., successes, total
            for context_key, (successes, total) in self.context_models.items():
                f.write(struct.pack(">I", len(context_key)))
                for crumb in context_key:
                    f.write(struct.pack(">I", crumb))
                f.write(struct.pack(">II", successes, total))

    def load_state(self) -> None:
        """Загрузить состояние иерархической модели"""
        # Загрузить групповую статистику
        if os.path.exists(self.group_file):
            with open(self.group_file, 'rb') as f:
                group_count = struct.unpack(">I", f.read(4))[0]
                self.group_stats.clear()
                for _ in range(group_count):
                    group, alpha, beta = struct.unpack(">Iff", f.read(12))
                    self.group_stats[group] = (alpha, beta)
        
        # Загрузить контекстные модели
        if os.path.exists(self.context_file):
            with open(self.context_file, 'rb') as f:
                context_count = struct.unpack(">I", f.read(4))[0]
                self.context_models.clear()
                for _ in range(context_count):
                    context_len = struct.unpack(">I", f.read(4))[0]
                    context_key = []
                    for _ in range(context_len):
                        crumb = struct.unpack(">I", f.read(4))[0]
                        context_key.append(crumb)
                    successes, total = struct.unpack(">II", f.read(8))
                    self.context_models[tuple(context_key)] = (successes, total)

    def print_stats(self) -> None:
        """Вывести статистику иерархической модели"""
        print(f"🏛️  Иерархическая Байесовская модель {self.name}:")
        print(f"   Групп: {len(self.group_stats)}")
        print(f"   Контекстных моделей: {len(self.context_models)}")
        print(f"   Иерархических предсказаний: {self.hierarchical_predictions}")
        print(f"   Успешных предсказаний: {self.hierarchical_successes}")
        if self.hierarchical_predictions > 0:
            accuracy = self.get_hierarchical_accuracy()
            print(f"   Точность: {accuracy:.2%}")