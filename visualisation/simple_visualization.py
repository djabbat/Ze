import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
import matplotlib.widgets as widgets
from datetime import datetime
import yaml

class Statistics:
    def __init__(self):
        self.reset_stats()
    
    def reset_stats(self):
        self.total_chunks = 0
        self.total_crumbs = 0
        self.match_history = []
        self.chunk_history = []
        self.last_update = datetime.now()

def load_config():
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            if 'processing' not in config:
                config['processing'] = {}
            return config
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return {'processing': {}}

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

def should_stop():
    return os.path.exists("data/stop.flag")

def calculate_match_ratio(current_matches, total_crumbs):
    if total_crumbs == 0:
        return 0.0
    return current_matches / total_crumbs

def main():
    Path("data").mkdir(exist_ok=True)
    config = load_config()
    
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    plt.suptitle("Crumb Processor Visualization", fontsize=14, y=0.98)
    
    # Настройка областей графиков
    ax_beg = plt.subplot2grid((2, 2), (0, 0))
    ax_inv = plt.subplot2grid((2, 2), (0, 1))
    ax_ratio = plt.subplot2grid((2, 2), (1, 0))
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
            
            # Обновление статистики
            beg_stats.total_chunks += 1
            inv_stats.total_chunks += 1
            
            beg_stats.total_crumbs += len(beg_data)
            inv_stats.total_crumbs += len(inv_data)
            
            current_beg_matches = np.sum(beg_data[:, 3] > 0) if len(beg_data) > 0 else 0
            current_inv_matches = np.sum(inv_data[:, 3] > 0) if len(inv_data) > 0 else 0
            
            beg_stats.match_history.append(current_beg_matches)
            inv_stats.match_history.append(current_inv_matches)
            
            beg_stats.chunk_history.append(beg_stats.total_chunks)
            inv_stats.chunk_history.append(inv_stats.total_chunks)
            
            beg_stats.last_update = datetime.now()
            inv_stats.last_update = datetime.now()
            
            # Очистка графиков
            ax_beg.clear()
            ax_inv.clear()
            ax_ratio.clear()
            ax_stats.clear()
            
            # График Beginning - количество счетчиков
            if len(beg_data) > 0:
                ax_beg.plot(beg_data[:, 2], 'b-', linewidth=0.8)
                ax_beg.set_title(f'Beginning Counters (Total: {len(beg_data)})')
                ax_beg.grid(True, linestyle=':', alpha=0.5)
            
            # График Inverse - количество счетчиков
            if len(inv_data) > 0:
                ax_inv.plot(inv_data[:, 2], 'm-', linewidth=0.8)
                ax_inv.set_title(f'Inverse Counters (Total: {len(inv_data)})')
                ax_inv.grid(True, linestyle=':', alpha=0.5)
            
            # График соотношения Matches/Crumbs
            if beg_stats.total_crumbs > 0 and inv_stats.total_crumbs > 0:
                beg_ratio = [m/beg_stats.total_crumbs for m in beg_stats.match_history]
                inv_ratio = [m/inv_stats.total_crumbs for m in inv_stats.match_history]
                
                ax_ratio.plot(beg_ratio, 'b-', label='Beginning')
                ax_ratio.plot(inv_ratio, 'm-', label='Inverse')
                ax_ratio.set_title('Matches/Crumbs Ratio')
                ax_ratio.legend(loc='upper left')
                ax_ratio.grid(True, linestyle=':', alpha=0.5)
            
            # Текстовая статистика
            ax_stats.axis('off')
            
            # Формируем текст статистики
            stats_text = f"Last Update: {datetime.now().strftime('%H:%M:%S')}\n\n"
            
            # Параметры конфигурации
            stats_text += "Configuration Parameters:\n"
            stats_text += f"crumb_size: {config['processing'].get('crumb_size', 2)}\n"
            stats_text += f"chunk_power: {config['processing'].get('chunk_power', 65536)}\n"
            stats_text += f"counter_value: {config['processing'].get('counter_value', 65536)}\n"
            stats_text += f"predict_increment: {config['processing'].get('predict_increment', 3)}\n"
            stats_text += f"actualization_value: {config['processing'].get('actualization_value', 0.99)}\n"
            stats_text += f"increment: {config['processing'].get('increment', 2)}\n"
            stats_text += f"filtration_value: {config['processing'].get('filtration_value', 100)}\n"
            stats_text += f"filtration_period: {config['processing'].get('filtration_period', 1000)}\n"
            stats_text += f"initial_id: {config['processing'].get('initial_id', 1)}\n\n"
            
            # Beginning статистика
            stats_text += "Beginning Processor:\n"
            stats_text += f"Total Crumbs: {beg_stats.total_crumbs}\n"
            stats_text += f"Current Matches: {current_beg_matches}\n"
            stats_text += f"Matches/Crumb: {calculate_match_ratio(current_beg_matches, beg_stats.total_crumbs):.4f}\n\n"
            
            # Inverse статистика
            stats_text += "Inverse Processor:\n"
            stats_text += f"Total Crumbs: {inv_stats.total_crumbs}\n"
            stats_text += f"Current Matches: {current_inv_matches}\n"
            stats_text += f"Matches/Crumb: {calculate_match_ratio(current_inv_matches, inv_stats.total_crumbs):.4f}"
            
            ax_stats.text(0.05, 0.95, stats_text, fontsize=8, family='monospace',
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