import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
import matplotlib.widgets as widgets

def safe_read_counters(filename):
    """Safe binary counter reader with .tmp filter"""
    if filename.endswith('.tmp'):
        print(f"Skipping temporary file: {filename}")
        return np.zeros((0, 2))
    
    attempts = 0
    min_size = 8  # At least one pair of uint32 (8 bytes)

    while attempts < 3:
        try:
            if not os.path.exists(filename) or os.path.getsize(filename) < min_size:
                return np.zeros((0, 2))

            with open(filename, 'rb') as f:
                data = np.fromfile(f, dtype=np.uint32)
                if len(data) < 2:
                    return np.zeros((0, 2))
                return data.reshape(-1, 2)

        except Exception as e:
            attempts += 1
            time.sleep(0.1)
    
    print(f"Failed to read {filename} after 3 attempts")
    return np.zeros((0, 2))

class Statistics:
    def __init__(self):
        self.total_chunks = 0
        self.total_predicts = 0
        self.matched_predicts = 0
        self.match_history = []
        self.chunk_history = []
    
    def update(self, counters, chunk_count):
        if len(counters) == 0:
            return
        
        matches = np.sum(counters[:, 0] == counters[:, 1])
        predicts = np.sum(counters[:, 1] % 2 == 0)
        
        self.total_chunks += chunk_count
        self.total_predicts += predicts
        self.matched_predicts += matches
        
        if predicts > 0:
            match_percent = (matches / predicts) * 100
            self.match_history.append(match_percent)
            self.chunk_history.append(self.total_chunks)
        
    def get_stats(self):
        stats = {
            'total_predicts': self.total_predicts,
            'matched_predicts': self.matched_predicts,
            'total_chunks': self.total_chunks,
            'match_percent': (self.matched_predicts / self.total_predicts * 100) if self.total_predicts else 0.0
        }
        return stats

def should_stop():
    return os.path.exists("data/stop.flag")

def main():
    # Удаляем старый флаг при запуске
    if os.path.exists("data/stop.flag"):
        os.remove("data/stop.flag")
    
    plt.ion()
    fig = plt.figure(figsize=(12, 8))
    ax_beg = plt.subplot2grid((2, 2), (0, 0))
    ax_inv = plt.subplot2grid((2, 2), (0, 1))
    ax_txt = plt.subplot2grid((2, 2), (1, 1))
    ax_plot = plt.subplot2grid((2, 2), (1, 0))

    beg_stats = Statistics()
    inv_stats = Statistics()

    # Управление
    should_stop_internal = False
    def stop(event):
        nonlocal should_stop_internal
        should_stop_internal = True

    def save(event):
        fig.savefig("visualization_snapshot.png")
        print("Saved figure to visualization_snapshot.png")

    stop_ax = plt.axes([0.8, 0.01, 0.1, 0.04])
    stop_btn = widgets.Button(stop_ax, 'Stop')
    stop_btn.on_clicked(stop)

    save_ax = plt.axes([0.65, 0.01, 0.1, 0.04])
    save_btn = widgets.Button(save_ax, 'Save PNG')
    save_btn.on_clicked(save)

    chunk_counter = 0
    print("Visualization started. Press 'Stop' button to end...")

    while not should_stop() and not should_stop_internal:
        chunk_counter += 1

        beg_data = safe_read_counters("data/beginning.bin")
        inv_data = safe_read_counters("data/inverse.bin")

        beg_stats.update(beg_data, 1)
        inv_stats.update(inv_data, 1)

        ax_beg.clear()
        ax_inv.clear()
        ax_txt.clear()
        ax_plot.clear()

        if len(beg_data) > 0:
            ax_beg.plot(beg_data[:, 1], 'b-')
            ax_beg.set_title(f'Beginning (Total: {len(beg_data)})')
            ax_beg.grid(True)

        if len(inv_data) > 0:
            ax_inv.plot(inv_data[:, 1], 'r-')
            ax_inv.set_title(f'Inverse (Total: {len(inv_data)})')
            ax_inv.grid(True)

        beg_s = beg_stats.get_stats()
        inv_s = inv_stats.get_stats()

        ax_txt.axis("off")
        text = f"""
        Beginning Processor:
        1. Total predictions made: {beg_s['total_predicts']}
        2. Predictions matched with Crumb: {beg_s['matched_predicts']}
        3. Match percentage: {beg_s['match_percent']:.2f}%
        4. Chunks processed: {beg_s['total_chunks']}

        Inverse Processor:
        1. Total predictions made: {inv_s['total_predicts']}
        2. Predictions matched with Crumb: {inv_s['matched_predicts']}
        3. Match percentage: {inv_s['match_percent']:.2f}%
        4. Chunks processed: {inv_s['total_chunks']}
        """
        ax_txt.text(0, 0.8, text, fontsize=10, verticalalignment='top', family='monospace')

        if beg_stats.match_history:
            ax_plot.plot(beg_stats.chunk_history, beg_stats.match_history, 'b-', label='Beginning')
        if inv_stats.match_history:
            ax_plot.plot(inv_stats.chunk_history, inv_stats.match_history, 'r-', label='Inverse')

        ax_plot.set_title("Prediction-Crumb Match Percentage")
        ax_plot.set_xlabel("Chunks Processed")
        ax_plot.set_ylabel("Match %")
        ax_plot.legend()
        ax_plot.grid(True)

        try:
            plt.pause(1.0)
        except:
            break

    print("Visualization ended.")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    main()