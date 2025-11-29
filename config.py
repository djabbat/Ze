# config.py
# Конфигурационные параметры для обработки данных

# Размеры данных
CHUNK_SIZE = 4096  # Размер chunk'а в байтах
CRUMB_SIZE = 2     # Размер crumb'а в байтах (2 байта = 16 бит)

# Параметры алгоритма
MAX_COUNTER_VALUE = 0xFFFFFFFF  # Максимальное значение для счетчиков
MAX_STATS_VALUE = 0xFFFFFFFF    # Максимальное значение для статистики
PREDICT_INCREMENT = 5  # Инкремент для зоны актуализации
INCREMENT = 1         # Инкремент для остальных зон
ACTUALIZATION_RATIO = 0.2  # Доля счетчиков в зоне актуализации (20%)

# Байесовские параметры
BAYES_ALPHA = 1.0     # Alpha параметр для априорного распределения
BAYES_BETA = 1.0      # Beta параметр для априорного распределения
CONFIDENCE_THRESHOLD = 0.7  # Порог уверенности для байесовского предсказания
MIN_OBSERVATIONS = 10  # Минимальное количество наблюдений для байесовского предсказания

# Пути к файлам
INPUT_FILE = "input/input.bin"
OUTPUT_DIR = "output"
STATS_DIR = "stats"