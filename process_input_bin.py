# process_input_bin.py
import struct
import os
from typing import List, Tuple, Dict, Optional
import subprocess
import sys
from config import *
from bayesian_predictor import BayesianPredictor

class Processor:
    def __init__(self, name: str):
        self.name = name
        self.bayesian = BayesianPredictor(name)
        
        # Счетчики для Crumb'ов
        self.counters: Dict[int, int] = {}  # key: crumb_value, value: counter
        
        # История для контекста иерархических моделей
        self.context_history: List[int] = []
        
        # Статистика
        self.total_matches_actualization = 0
        self.total_matches_others = 0
        self.total_first_prediction_hits = 0
        self.total_crumbs = 0
        self.total_bayesian_hits = 0
        
        # Файлы для сохранения
        self.state_file = os.path.join(OUTPUT_DIR, f"{name}.bin")  # Основные данные в output
        self.stats_file = os.path.join(STATS_DIR, f"{name}_matches.bin")  # Статистика в stats

    def process_crumb(self, crumb: int) -> None:
        """Обработка одного Crumb'а с улучшенным байесовским предсказанием"""
        self._increment_total_crumbs()
        
        # Получить текущий контекст для иерархического анализа
        current_context = self._get_current_context()
        
        # Шаг 1: Попробовать улучшенное предсказание с иерархическими моделями
        prediction_result = self._try_enhanced_prediction(crumb, current_context)
        if prediction_result is not None:
            predicted_crumb, probability, confidence, method = prediction_result
            self.total_bayesian_hits += 1
            
            # Обновить статистику с учетом контекста
            self.bayesian.record_prediction_result(predicted_crumb, crumb, current_context)
            
            # Также обрабатываем через стандартный алгоритм
            self._standard_processing(crumb, is_predicted=True)
            
            # Обновить историю контекста
            self._update_context_history(crumb)
            return
        
        # Шаг 2: Стандартная обработка
        self._standard_processing(crumb, is_predicted=False)
        
        # Обновить базовую статистику
        self.bayesian.update_stats(crumb, True)
        self.bayesian.update_context(crumb)
        
        # Обновить историю контекста
        self._update_context_history(crumb)

    def _try_enhanced_prediction(self, current_crumb: int, context: List[int]) -> Optional[Tuple]:
        """Улучшенная попытка предсказания с иерархическими моделями"""
        available_crumbs = self._get_available_crumbs()
        if not available_crumbs:
            return None
            
        return self.bayesian.predict_next(context, available_crumbs)

    def _get_available_crumbs(self) -> List[int]:
        """Получить список доступных Crumb'ов для предсказания"""
        # Используем топ-N самых частых счетчиков
        sorted_counters = sorted(self.counters.items(), key=lambda x: x[1], reverse=True)
        return [crumb for crumb, count in sorted_counters[:min(50, len(sorted_counters))]]

    def _get_current_context(self) -> List[int]:
        """Получить текущий контекст для иерархического анализа"""
        # Возвращаем последние N Crumb'ов как контекст
        return self.context_history[-CONTEXT_DEPTH:] if self.context_history else []

    def _update_context_history(self, crumb: int) -> None:
        """Обновить историю контекста"""
        self.context_history.append(crumb)
        # Ограничить глубину контекста для экономии памяти
        if len(self.context_history) > CONTEXT_DEPTH * 3:
            self.context_history = self.context_history[-CONTEXT_DEPTH * 2:]

    def _standard_processing(self, crumb: int, is_predicted: bool) -> None:
        """Стандартная обработка Crumb'а"""
        # Отсортировать счетчики
        sorted_counters = sorted(self.counters.items(), key=lambda x: x[1], reverse=True)
        
        # Определяем зону актуализации
        actualization_count = max(1, int(len(sorted_counters) * ACTUALIZATION_RATIO))
        actualization_zone = dict(sorted_counters[:actualization_count])
        other_zone = dict(sorted_counters[actualization_count:])
        
        # Проверить зону актуализации
        if crumb in actualization_zone:
            self._increment_stat_actualization()
            self._increment_counter(crumb, PREDICT_INCREMENT)
            if self.total_crumbs == 1:
                self._increment_first_prediction()
            return
        
        # Проверить остальные счетчики
        if crumb in other_zone:
            self._increment_stat_others()
            self._increment_counter(crumb, INCREMENT)
            return
        
        # Создать новый счетчик
        self.counters[crumb] = 1

    def _increment_counter(self, crumb: int, increment: int) -> None:
        """Инкремент счетчика с проверкой на максимальное значение"""
        current_value = self.counters.get(crumb, 0)
        new_value = current_value + increment
        
        if new_value > MAX_COUNTER_VALUE:
            print(f"⚠️  Достигнут максимальный предел счетчика {crumb}: {current_value} -> {new_value}")
            print(f"🔧 Деление всех счетчиков на 2...")
            self._reset_counters()
            self.counters[crumb] = current_value // 2 + increment
        else:
            self.counters[crumb] = new_value

    def _reset_counters(self) -> None:
        """Сброс счетчиков - деление всех значений на 2"""
        reset_count = 0
        for key in list(self.counters.keys()):
            old_value = self.counters[key]
            new_value = max(1, self.counters[key] // 2)
            self.counters[key] = new_value
            if old_value != new_value:
                reset_count += 1
        
        if reset_count > 0:
            print(f"🔄 Сброшено {reset_count} счетчиков (деление на 2)")

    def _increment_stat_actualization(self) -> None:
        """Инкремент статистики для зоны актуализации"""
        if self.total_matches_actualization >= MAX_STATS_VALUE:
            self._reset_stats()
        self.total_matches_actualization += 1

    def _increment_stat_others(self) -> None:
        """Инкремент статистики для остальных зон"""
        if self.total_matches_others >= MAX_STATS_VALUE:
            self._reset_stats()
        self.total_matches_others += 1

    def _increment_first_prediction(self) -> None:
        """Инкремент статистики первого предсказания"""
        if self.total_first_prediction_hits >= MAX_STATS_VALUE:
            self._reset_stats()
        self.total_first_prediction_hits += 1

    def _increment_total_crumbs(self) -> None:
        """Инкремент общего количества crumb'ов"""
        if self.total_crumbs >= MAX_STATS_VALUE:
            self._reset_stats()
        self.total_crumbs += 1

    def _reset_stats(self) -> None:
        """Сброс всей статистики - деление всех значений на 2"""
        print(f"🔄 Сброс статистики (деление на 2):")
        print(f"   total_matches_actualization: {self.total_matches_actualization} -> {self.total_matches_actualization // 2}")
        print(f"   total_matches_others: {self.total_matches_others} -> {self.total_matches_others // 2}")
        print(f"   total_first_prediction_hits: {self.total_first_prediction_hits} -> {self.total_first_prediction_hits // 2}")
        print(f"   total_crumbs: {self.total_crumbs} -> {self.total_crumbs // 2}")
        print(f"   total_bayesian_hits: {self.total_bayesian_hits} -> {self.total_bayesian_hits // 2}")
        
        self.total_matches_actualization = max(1, self.total_matches_actualization // 2)
        self.total_matches_others = max(1, self.total_matches_others // 2)
        self.total_first_prediction_hits = max(1, self.total_first_prediction_hits // 2)
        self.total_crumbs = max(1, self.total_crumbs // 2)
        self.total_bayesian_hits = max(1, self.total_bayesian_hits // 2)

    def save_state(self) -> None:
        """Сохранение состояния"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(STATS_DIR, exist_ok=True)
        
        # Сохраняем счетчики в output
        with open(self.state_file, 'wb') as f:
            for key, value in self.counters.items():
                f.write(struct.pack(">II", key, value))
        
        # Сохраняем статистику в stats
        with open(self.stats_file, 'wb') as f:
            stats = [
                (0, self.total_matches_actualization),
                (1, self.total_matches_others),
                (2, self.total_first_prediction_hits),
                (3, self.total_crumbs),
                (4, self.total_bayesian_hits)
            ]
            for key, value in stats:
                f.write(struct.pack(">II", key, value))
        
        # Сохраняем байесовскую статистику (включая иерархические модели)
        self.bayesian.save_state()

    def load_state(self) -> None:
        """Загрузка состояния"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(STATS_DIR, exist_ok=True)
        
        # Загружаем счетчики из output
        if os.path.exists(self.state_file):
            self.counters.clear()
            with open(self.state_file, 'rb') as f:
                data = f.read()
                for i in range(0, len(data), 8):
                    key, value = struct.unpack(">II", data[i:i+8])
                    self.counters[key] = value
        
        # Загружаем статистику из stats
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'rb') as f:
                data = f.read()
                for i in range(0, len(data), 8):
                    key, value = struct.unpack(">II", data[i:i+8])
                    if key == 0:
                        self.total_matches_actualization = value
                    elif key == 1:
                        self.total_matches_others = value
                    elif key == 2:
                        self.total_first_prediction_hits = value
                    elif key == 3:
                        self.total_crumbs = value
                    elif key == 4:
                        self.total_bayesian_hits = value
        
        # Загружаем байесовскую статистику (включая иерархические модели)
        self.bayesian.load_state()


def read_chunk(file, chunk_size: int) -> bytes:
    """Чтение chunk'а из файла"""
    chunk = file.read(chunk_size)
    return chunk if chunk else b''


def process_chunk_beginning(chunk: bytes, processor: Processor) -> None:
    """Обработка chunk'а с начала"""
    for i in range(0, len(chunk), CRUMB_SIZE):
        crumb_bytes = chunk[i:i + CRUMB_SIZE]
        # Дополняем нулями если нужно
        if len(crumb_bytes) < CRUMB_SIZE:
            crumb_bytes += b'\x00' * (CRUMB_SIZE - len(crumb_bytes))
        
        # Конвертируем в число (big-endian)
        crumb_value = int.from_bytes(crumb_bytes, byteorder='big')
        processor.process_crumb(crumb_value)


def process_chunk_inverse(chunk: bytes, processor: Processor) -> None:
    """Обработка chunk'а с конца"""
    # Обрабатываем байты в обратном порядке
    reversed_chunk = chunk[::-1]
    
    for i in range(0, len(reversed_chunk), CRUMB_SIZE):
        crumb_bytes = reversed_chunk[i:i + CRUMB_SIZE]
        # Дополняем нулями если нужно
        if len(crumb_bytes) < CRUMB_SIZE:
            crumb_bytes += b'\x00' * (CRUMB_SIZE - len(crumb_bytes))
        
        # Конвертируем в число (big-endian)
        crumb_value = int.from_bytes(crumb_bytes, byteorder='big')
        processor.process_crumb(crumb_value)


def run_audio_stream():
    """Запуск аудио модуля после завершения основной обработки"""
    print("\n" + "="*50)
    print("Запуск аудио модуля...")
    print("="*50)
    
    try:
        # Проверяем существует ли файл audio_stream.py
        if os.path.exists("audio_stream.py"):
            subprocess.run([sys.executable, "audio_stream.py"])
        else:
            print("Файл audio_stream.py не найден")
    except Exception as e:
        print(f"Ошибка запуска аудио модуля: {e}")


def main():
    # Создаем директории если не существуют
    os.makedirs("input", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    os.makedirs(HIERARCHICAL_DIR, exist_ok=True)
    
    # Проверяем существование входного файла
    if not os.path.exists(INPUT_FILE):
        print(f"Ошибка: Входной файл {INPUT_FILE} не найден!")
        print("Создайте файл input/input.bin с тестовыми данными")
        return
    
    # Выводим текущую конфигурацию
    print("Текущая конфигурация:")
    print(f"  CHUNK_SIZE: {CHUNK_SIZE}")
    print(f"  CRUMB_SIZE: {CRUMB_SIZE}")
    print(f"  MAX_COUNTER_VALUE: {MAX_COUNTER_VALUE} (0x{MAX_COUNTER_VALUE:08X})")
    print(f"  MAX_STATS_VALUE: {MAX_STATS_VALUE} (0x{MAX_STATS_VALUE:08X})")
    print(f"  PREDICT_INCREMENT: {PREDICT_INCREMENT}")
    print(f"  INCREMENT: {INCREMENT}")
    print(f"  ACTUALIZATION_RATIO: {ACTUALIZATION_RATIO}")
    print(f"  Базовые байесовские параметры:")
    print(f"    ALPHA: {BAYES_ALPHA}, BETA: {BAYES_BETA}")
    print(f"    CONFIDENCE_THRESHOLD: {CONFIDENCE_THRESHOLD}")
    print(f"    MIN_OBSERVATIONS: {MIN_OBSERVATIONS}")
    print(f"  Иерархические параметры:")
    print(f"    Включено: {HIERARCHICAL_ENABLED}")
    print(f"    GROUP_SIZE: {GROUP_SIZE}")
    print(f"    CONTEXT_DEPTH: {CONTEXT_DEPTH}")
    print(f"    ALPHA_PRIOR: {HIERARCHICAL_ALPHA_PRIOR}")
    print(f"    BETA_PRIOR: {HIERARCHICAL_BETA_PRIOR}")
    print()
    
    # Инициализация процессоров
    beginning_processor = Processor("begin")
    inverse_processor = Processor("inverse")
    
    # Загружаем предыдущее состояние (если есть)
    beginning_processor.load_state()
    inverse_processor.load_state()
    
    # Обработка файла
    try:
        with open(INPUT_FILE, 'rb') as file:
            chunk_count = 0
            
            while True:
                chunk = read_chunk(file, CHUNK_SIZE)
                if not chunk:
                    break
                
                print(f"Обработка chunk #{chunk_count + 1}, размер: {len(chunk)} байт")
                
                # Обработка beginning процессором
                process_chunk_beginning(chunk, beginning_processor)
                
                # Обработка inverse процессором
                process_chunk_inverse(chunk, inverse_processor)
                
                chunk_count += 1
                
                # Периодически сохраняем состояние
                if chunk_count % 10 == 0:
                    beginning_processor.save_state()
                    inverse_processor.save_state()
                    
                    # Выводим информацию о максимальных счетчиках и статистике
                    if beginning_processor.counters:
                        max_begin = max(beginning_processor.counters.values())
                        print(f"  Beginning: счетчиков={len(beginning_processor.counters)}, макс={max_begin}")
                    if inverse_processor.counters:
                        max_inverse = max(inverse_processor.counters.values())
                        print(f"  Inverse: счетчиков={len(inverse_processor.counters)}, макс={max_inverse}")
        
        # Финальное сохранение
        beginning_processor.save_state()
        inverse_processor.save_state()
        
        print("\n" + "="*50)
        print("ОБРАБОТКА ЗАВЕРШЕНА!")
        print("="*50)
        print(f"Обработано chunk'ов: {chunk_count}")
        print(f"Общее количество Crumb'ов: {beginning_processor.total_crumbs + inverse_processor.total_crumbs}")
        
        # Вывод статистики процессоров
        print("\n📊 СТАТИСТИКА ПРОЦЕССОРОВ:")
        print("\nBeginning процессор:")
        print(f"  Обработано Crumb'ов: {beginning_processor.total_crumbs}")
        print(f"  Уникальных счетчиков: {len(beginning_processor.counters)}")
        print(f"  Совпадения в зоне актуализации: {beginning_processor.total_matches_actualization}")
        print(f"  Совпадения в остальных зонах: {beginning_processor.total_matches_others}")
        print(f"  Попадания с первого предсказания: {beginning_processor.total_first_prediction_hits}")
        print(f"  Байесовские попадания: {beginning_processor.total_bayesian_hits}")
        
        print("\nInverse процессор:")
        print(f"  Обработано Crumb'ов: {inverse_processor.total_crumbs}")
        print(f"  Уникальных счетчиков: {len(inverse_processor.counters)}")
        print(f"  Совпадения в зоне актуализации: {inverse_processor.total_matches_actualization}")
        print(f"  Совпадения в остальных зонах: {inverse_processor.total_matches_others}")
        print(f"  Попадания с первого предсказания: {inverse_processor.total_first_prediction_hits}")
        print(f"  Байесовские попадания: {inverse_processor.total_bayesian_hits}")
        
        # Вывод байесовской статистики
        print("\n" + "="*50)
        print("БАЙЕСОВСКАЯ СТАТИСТИКА:")
        print("="*50)
        beginning_processor.bayesian.print_stats()
        print()
        inverse_processor.bayesian.print_stats()
        
        # Общая статистика эффективности
        print("\n" + "="*50)
        print("ОБЩАЯ ЭФФЕКТИВНОСТЬ:")
        print("="*50)
        total_crumbs = beginning_processor.total_crumbs + inverse_processor.total_crumbs
        total_bayesian_hits = beginning_processor.total_bayesian_hits + inverse_processor.total_bayesian_hits
        
        if total_crumbs > 0:
            bayesian_efficiency = total_bayesian_hits / total_crumbs
            print(f"Общая эффективность байесовских предсказаний: {bayesian_efficiency:.2%}")
        
        print(f"\n💾 ФАЙЛЫ СОХРАНЕНЫ:")
        print(f"  Основные данные в '{OUTPUT_DIR}':")
        print(f"    - begin.bin, inverse.bin")
        print(f"  Статистика в '{STATS_DIR}':")
        print(f"    - begin_matches.bin, inverse_matches.bin")
        print(f"    - begin_bayes.bin, inverse_bayes.bin")
        print(f"  Иерархические модели в '{HIERARCHICAL_DIR}':")
        print(f"    - begin_groups.bin, inverse_groups.bin")
        print(f"    - begin_context.bin, inverse_context.bin")
        
        # Запуск аудио модуля в конце
        run_audio_stream()
        
    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()