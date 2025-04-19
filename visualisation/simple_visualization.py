import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
import matplotlib.widgets as widgets
from datetime import datetime

class Statistics:
    def __init__(self):
        self.reset_stats()
    
    def reset_stats(self):
        self.total_chunks = 0
        self.total_crumbs = 0
        self.match_history = []
        self.chunk_history = []
        self.last_update = datetime.now()
    
    def update(self, counters):
        if len(counters) == 0:
            return
        
        self.total_crumbs += len(counters)
        self.match_history.append(np.sum(counters[:, 3] > 0))
        self.chunk_history.append(self.total_chunks)
        self.last_update = datetime.now()

def safe_read_counters(filename):
    try:
        if not os.path.exists(filename) or os.path.getsize(filename) < 4:
            return np.zeros((0, 4))
        
        with open(filename, 'rb') as f:
            count = np.fromfile(f, dtype=np.uint32, count=1)[0]
            if count == 0:
                return np.zeros((0, 4))
            
            dtype = np.dtype([
                ('id', np.uint32),
                ('data', 'V4'),
                ('value', np.uint32),
                ('matches', np.uint32)
            ])
            data = np.fromfile(f, dtype=dtype, count=count)
            
            result = np.zeros((len(data), 4))
            result[:, 0] = data['id']
            result[:, 1] = data['data'].view(np.uint32)
            result[:, 2] = data['value']
            result[:, 3] = data['matches']
            
            return result
            
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")
        return np.zeros((0, 4))

def safe_read_matches(filename):
    try:
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            return np.zeros((0, 2))
        
        with open(filename, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint32)
            return data.reshape(-1, 2)
            
    except Exception as e:
        print(f"Error reading matches file {filename}: {str(e)}")
        return np.zeros((0, 2))

def get_top_counters(counters, n=3):
    if len(counters) == 0:
        return []
    
    sorted_indices = np.argsort(-counters[:, 2])
    top_indices = sorted_indices[:min(n, len(counters))]
    
    return [(int(counters[i, 0]), int(counters[i, 2])) for i in top_indices]

def should_stop():
    return os.path.exists("data/stop.flag")

def main():
    Path("data").mkdir(exist_ok=True)
    
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    plt.suptitle("Crumb Processor Visualization", fontsize=14, y=0.98)
    
    # Настройка областей графиков
    ax_beg = plt.subplot2grid((2, 2), (0, 0))
    ax_inv = plt.subplot2grid((2, 2), (0, 1))
    ax_plot = plt.subplot2grid((2, 2), (1, 0))
    ax_stats = plt.subplot2grid((2, 2), (1, 1))
    
    # Настройка отступов
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9, hspace=0.5, wspace=0.5)
    
    # Кнопки управления
    stop_ax = plt.axes([0.82, 0.02, 0.08, 0.05])
    save_ax = plt.axes([0.72, 0.02, 0.08, 0.05])
    reset_ax = plt.axes([0.62, 0.02, 0.08, 0.05])
    
    stop_btn = widgets.Button(stop_ax, 'Stop', color='tomato')
    save_btn = widgets.Button(save_ax, 'Save', color='lightgreen')
    reset_btn = widgets.Button(reset_ax, 'Reset', color='lightblue')
    
    beg_stats = Statistics()
    inv_stats = Statistics()
    
    def on_stop(event):
        with open("data/stop.flag", "w") as f:
            f.write("stop")
    
    def on_save(event):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visualization_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {filename}")
    
    def on_reset(event):
        beg_stats.reset_stats()
        inv_stats.reset_stats()
        print("Statistics reset")
    
    stop_btn.on_clicked(on_stop)
    save_btn.on_clicked(on_save)
    reset_btn.on_clicked(on_reset)
    
    print("Visualization started. Controls: Stop | Save | Reset")
    
    try:
        while not should_stop():
            # Чтение данных
            beg_data = safe_read_counters("data/beginning.bin")
            inv_data = safe_read_counters("data/inverse.bin")
            beg_matches = safe_read_matches("data/beginning_matches.bin")
            inv_matches = safe_read_matches("data/inverse_matches.bin")
            
            # Обновление статистики
            beg_stats.total_chunks += 1
            inv_stats.total_chunks += 1
            beg_stats.update(beg_data)
            inv_stats.update(inv_data)
            
            # Очистка графиков
            ax_beg.clear()
            ax_inv.clear()
            ax_plot.clear()
            ax_stats.clear()
            
            # График Beginning
            if len(beg_data) > 0:
                ax_beg.plot(beg_data[:, 2], 'b-', linewidth=0.8)
                matches = np.where(beg_data[:, 3] > 0)
                ax_beg.plot(matches, beg_data[matches, 2], 'ro', markersize=3)
                ax_beg.set_title(f'Beginning (N={len(beg_data)})')
                ax_beg.grid(True, linestyle=':', alpha=0.5)
            
            # График Inverse
            if len(inv_data) > 0:
                ax_inv.plot(inv_data[:, 2], 'm-', linewidth=0.8)
                matches = np.where(inv_data[:, 3] > 0)
                ax_inv.plot(matches, inv_data[matches, 2], 'bo', markersize=3)
                ax_inv.set_title(f'Inverse (N={len(inv_data)})')
                ax_inv.grid(True, linestyle=':', alpha=0.5)
            
            # График истории совпадений
            if beg_stats.chunk_history:
                ax_plot.plot(beg_stats.chunk_history, beg_stats.match_history, 'b-', label='Beginning')
                ax_plot.plot(inv_stats.chunk_history, inv_stats.match_history, 'm-', label='Inverse')
                ax_plot.set_title('Matches History')
                ax_plot.legend(loc='upper left')
                ax_plot.grid(True, linestyle=':', alpha=0.5)
            
            # Текстовая статистика
            ax_stats.axis('off')
            
            # Получаем топ-3 счетчиков
            beg_top = get_top_counters(beg_data, 3)
            inv_top = get_top_counters(inv_data, 3)
            
            # Формируем текст статистики
            stats_text = f"Last Update: {datetime.now().strftime('%H:%M:%S')}\n\n"
            
            # Beginning статистика
            stats_text += "Beginning Processor:\n"
            stats_text += f"Chunks: {beg_stats.total_chunks}\n"
            stats_text += f"Total Crumbs: {beg_stats.total_crumbs}\n"
            stats_text += f"Current Matches: {beg_stats.match_history[-1] if beg_stats.match_history else 0}\n"
            
            if len(beg_matches) > 0:
                stats_text += f"Last Match: ID={beg_matches[-1][0]} (V={beg_matches[-1][1]})\n"
            
            stats_text += "\nTop Counters:\n"
            for i, (id_val, val) in enumerate(beg_top, 1):
                stats_text += f"{i}. ID={id_val} (V={val})\n"
            
            # Inverse статистика
            stats_text += "\nInverse Processor:\n"
            stats_text += f"Chunks: {inv_stats.total_chunks}\n"
            stats_text += f"Total Crumbs: {inv_stats.total_crumbs}\n"
            stats_text += f"Current Matches: {inv_stats.match_history[-1] if inv_stats.match_history else 0}\n"
            
            if len(inv_matches) > 0:
                stats_text += f"Last Match: ID={inv_matches[-1][0]} (V={inv_matches[-1][1]})\n"
            
            stats_text += "\nTop Counters:\n"
            for i, (id_val, val) in enumerate(inv_top, 1):
                stats_text += f"{i}. ID={id_val} (V={val})"
            
            ax_stats.text(0.05, 0.95, stats_text, fontsize=9, family='monospace',
                        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
            
            plt.pause(1.0)
            
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    finally:
        plt.ioff()
        plt.close()
        if os.path.exists("data/stop.flag"):
            os.remove("data/stop.flag")

if __name__ == "__main__":
    main()