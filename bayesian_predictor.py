# bayesian_predictor.py
import struct
import os
import math
from typing import Dict, Tuple, List, Optional
from config import *

class BayesianPredictor:
    """Отдельный байесовский предсказатель"""
    
    def __init__(self, name: str):
        self.name = name
        self.alpha = BAYES_ALPHA
        self.beta = BAYES_BETA
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.min_observations = MIN_OBSERVATIONS
        
        # Байесовская статистика: crumb -> (successes, total_attempts)
        self.stats: Dict[int, Tuple[int, int]] = {}
        
        # Файл для сохранения байесовской статистики (в папке stats)
        self.stats_file = os.path.join(STATS_DIR, f"{name}_bayes.bin")
        
        # Статистика работы байесовского предсказателя
        self.total_predictions = 0
        self.successful_predictions = 0
        self.confident_predictions = 0

    def update_stats(self, crumb: int, success: bool) -> None:
        """Обновление статистики для конкретного Crumb'а"""
        if crumb not in self.stats:
            self.stats[crumb] = (0, 0)
            
        successes, total = self.stats[crumb]
        if success:
            successes += 1
        total += 1
        
        # Проверка на максимальное значение
        if total >= MAX_STATS_VALUE or successes >= MAX_STATS_VALUE:
            successes = max(1, successes // 2)
            total = max(1, total // 2)
            
        self.stats[crumb] = (successes, total)

    def calculate_posterior(self, successes: int, total: int) -> float:
        """Вычисление апостериорной вероятности по Beta распределению"""
        if total == 0:
            return 0.0
            
        posterior_alpha = self.alpha + successes
        posterior_beta = self.beta + (total - successes)
        
        return posterior_alpha / (posterior_alpha + posterior_beta)

    def calculate_confidence(self, successes: int, total: int) -> float:
        """Вычисление уверенности в предсказании"""
        if total < self.min_observations:
            return 0.0
            
        probability = self.calculate_posterior(successes, total)
        
        # Вычисляем дисперсию Beta распределения
        posterior_alpha = self.alpha + successes
        posterior_beta = self.beta + (total - successes)
        sum_ab = posterior_alpha + posterior_beta
        
        variance = (posterior_alpha * posterior_beta) / (sum_ab ** 2 * (sum_ab + 1))
        
        # Уверенность обратно пропорциональна стандартному отклонению
        confidence = 1.0 - math.sqrt(variance) * 2
        return max(0.0, min(1.0, confidence))

    def predict_next(self, current_context: List[int], available_crumbs: List[int]) -> Optional[Tuple[int, float, float]]:
        """Предсказание следующего Crumb'а"""
        self.total_predictions += 1
        
        if len(self.stats) < self.min_observations:
            return None
            
        best_crumb = None
        best_confidence = 0.0
        best_probability = 0.0
        
        for crumb in available_crumbs:
            if crumb in self.stats:
                successes, total = self.stats[crumb]
                confidence = self.calculate_confidence(successes, total)
                probability = self.calculate_posterior(successes, total)
                
                if (confidence >= self.confidence_threshold and 
                    probability > 0.5 and 
                    confidence > best_confidence):
                    best_confidence = confidence
                    best_probability = probability
                    best_crumb = crumb
        
        if best_crumb is not None:
            self.confident_predictions += 1
            return (best_crumb, best_probability, best_confidence)
            
        return None

    def record_prediction_result(self, predicted_crumb: int, actual_crumb: int) -> None:
        """Запись результата предсказания"""
        success = (predicted_crumb == actual_crumb)
        if success:
            self.successful_predictions += 1
        
        # Обновляем статистику для предсказанного Crumb'а
        self.update_stats(predicted_crumb, success)
        
        # Также обновляем статистику для фактического Crumb'а
        self.update_stats(actual_crumb, True)

    def get_prediction_accuracy(self) -> float:
        """Получить точность предсказаний"""
        if self.confident_predictions == 0:
            return 0.0
        return self.successful_predictions / self.confident_predictions

    def save_state(self) -> None:
        """Сохранение байесовской статистики в папку stats"""
        os.makedirs(STATS_DIR, exist_ok=True)
        
        with open(self.stats_file, 'wb') as f:
            # Сохраняем общую статистику
            f.write(struct.pack(">III", 
                               self.total_predictions,
                               self.successful_predictions,
                               self.confident_predictions))
            
            # Сохраняем статистику по Crumb'ам
            for crumb, (successes, total) in self.stats.items():
                f.write(struct.pack(">III", crumb, successes, total))

    def load_state(self) -> None:
        """Загрузка байесовской статистики из папки stats"""
        if not os.path.exists(self.stats_file):
            return
            
        try:
            with open(self.stats_file, 'rb') as f:
                # Загружаем общую статистику
                header_data = f.read(12)
                if len(header_data) == 12:
                    self.total_predictions, self.successful_predictions, self.confident_predictions = \
                        struct.unpack(">III", header_data)
                
                # Загружаем статистику по Crumb'ам
                self.stats.clear()
                while True:
                    data = f.read(12)
                    if not data or len(data) < 12:
                        break
                    crumb, successes, total = struct.unpack(">III", data)
                    self.stats[crumb] = (successes, total)
                    
        except Exception as e:
            print(f"Ошибка загрузки байесовской статистики: {e}")

    def print_stats(self) -> None:
        """Вывод статистики байесовского предсказателя"""
        print(f"📊 Байесовский предсказатель {self.name}:")
        print(f"   Всего предсказаний: {self.total_predictions}")
        print(f"   Уверенных предсказаний: {self.confident_predictions}")
        print(f"   Успешных предсказаний: {self.successful_predictions}")
        if self.confident_predictions > 0:
            accuracy = self.get_prediction_accuracy()
            print(f"   Точность: {accuracy:.2%}")
        print(f"   Отслеживаемых Crumb'ов: {len(self.stats)}")