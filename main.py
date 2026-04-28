import sys
import os
import numpy as np
import pandas as pd
import scipy.io
import scipy.stats as stats
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QComboBox, 
                             QLabel, QMessageBox, QLineEdit, QTableWidget, 
                             QTableWidgetItem, QTabWidget, QHeaderView, QAbstractItemView)
from PyQt6.QtCore import Qt

class DataManager:
    def __init__(self):
        self.raw_data = None
        self.fs_vector = None
        # Store individual rates
        self.fs_n = self.fs_s = self.fs_t = 1.0
        self.titles = []
        self.datastart = []
        self.dataend = []
        self.needle_emg = None
        self.surface_emg = None
        self.torque = None
        self.spike_df = None 
        self.window_list = [] # Stores ranges in seconds
        self.toolbox_path = ""

    def load_mat(self, filepath):
        try:
            mat = scipy.io.loadmat(filepath)
            # Fetch the sampling rate vector
            self.fs_vector = mat['samplerate'].flatten().astype(float)
            self.raw_data = mat['data'].flatten()
            self.datastart = mat['datastart'].flatten()
            self.dataend = mat['dataend'].flatten()
            self.titles = [str(t[0]) if isinstance(t, (np.ndarray, list)) else str(t) for t in mat['titles'].flatten()]
            return True
        except: return False

    def load_ann(self, filepath):
        try:
            self.spike_df = pd.read_csv(filepath, sep=None, engine='python', header=None)
            self.spike_df.columns = ['timestamp', 'mu_id']
            self.spike_df['mu_id'] = self.spike_df['mu_id'].astype(str)
            self.spike_df['status'] = 'Valid'
            return True
        except: return False

    def map_channels(self, n_idx, s_idx, t_idx):
        try:
            self.fs_n = self.fs_vector[n_idx]
            self.fs_s = self.fs_vector[s_idx]
            self.fs_t = self.fs_vector[t_idx]
            self.needle_emg = self.raw_data[int(self.datastart[n_idx])-1 : int(self.dataend[n_idx])]
            self.surface_emg = self.raw_data[int(self.datastart[s_idx])-1 : int(self.dataend[s_idx])]
            self.torque = self.raw_data[int(self.datastart[t_idx])-1 : int(self.dataend[t_idx])]
            return True
        except: return False

    def update_mask(self, window_list):
        self.window_list = window_list

    def get_valid_spikes(self, mu_id):
        if self.spike_df is None: return np.array([])
        mask = (self.spike_df['mu_id'] == str(mu_id)) & (self.spike_df['status'] == 'Valid')
        all_ts = self.spike_df[mask]['timestamp'].values
        valid_ts = [t for t in all_ts if any(start <= t <= end for start, end in self.window_list)]
        return np.array(valid_ts)

    def get_template(self, mu_id, chan_type='needle'):
        ts = self.get_valid_spikes(mu_id)
        if len(ts) == 0: return None
        data = self.needle_emg if chan_type == 'needle' else self.surface_emg
        fs = self.fs_n if chan_type == 'needle' else self.fs_s
        half_win = int(np.rint(0.005 * fs))
        segments = [data[int(np.rint(t*fs))-half_win : int(np.rint(t*fs))+half_win] for t in ts 
                    if int(np.rint(t*fs))-half_win >= 0 and int(np.rint(t*fs))+half_win < len(data)]
        return np.mean(segments, axis=0) if segments else None

    def calculate_ptp_for_timestamps(self, ts_array):
        if self.needle_emg is None: return np.zeros(len(ts_array))
        half_win = int(np.rint(0.002 * self.fs_n))
        ptps = [np.ptp(self.needle_emg[max(0, int(np.rint(t*self.fs_n))-half_win) : 
                       min(len(self.needle_emg), int(np.rint(t*self.fs_n))+half_win)]) 
                for t in ts_array]
        return np.array(ptps)

    def merge_mu(self, keep_id, merge_id):
        self.spike_df.loc[self.spike_df['mu_id'] == str(merge_id), 'mu_id'] = str(keep_id)

    def split_mu(self, old_id, threshold, ts_to_split, ptps):
        mu_mask = (self.spike_df['mu_id'] == str(old_id))
        for t, ptp in zip(ts_to_split, ptps):
            match = self.spike_df[mu_mask & (np.isclose(self.spike_df['timestamp'], t))]
            if not match.empty:
                self.spike_df.at[match.index[0], 'mu_id'] = f"{old_id}_High" if ptp >= threshold else f"{old_id}_Low"

    def get_spike_segment(self, timestamp):
        """Returns a 10ms segment around a timestamp for the overlay plot."""
        half_win = int(np.rint(0.005 * self.fs_n)) # 5ms each side = 10ms total
        center_sample = int(np.rint(timestamp * self.fs_n))
        start, end = center_sample - half_win, center_sample + half_win
        if start >= 0 and end < len(self.needle_emg):
            return self.needle_emg[start:end]
        return None
    
    def get_spike_segment(self, timestamp):
        """Returns a 10ms segment around a timestamp for the overlay plot."""
        half_win = int(np.rint(0.005 * self.fs_n)) # 5ms each side
        center_sample = int(np.rint(timestamp * self.fs_n))
        start, end = center_sample - half_win, center_sample + half_win
        if start >= 0 and end < len(self.needle_emg):
            return self.needle_emg[start:end]
        return None

    def calculate_nmse(self, waveform, template):
        """Calculates Normalized Mean Squared Error."""
        if waveform is None or template is None or len(waveform) != len(template): 
            return 1.0
        mse = np.mean((waveform - template)**2)
        energy = np.mean(template**2)
        return mse / energy if energy > 0 else 1.0
    
    def calculate_pnr(self, mu_id):
        """Calculates Pulse-to-Noise Ratio (dB) for a specific MU."""
        temp = self.get_template(mu_id)
        if temp is None: return 0
        ptp = np.ptp(temp)
        # Estimate noise floor from the first 500ms of needle data
        noise_floor = np.std(self.needle_emg[:int(0.5 * self.fs_n)])
        return 20 * np.log10(ptp / noise_floor) if noise_floor > 0 and ptp > 0 else 0

    def get_recruitment_info(self, mu_id):
        """Returns mean torque values over the first 5 and last 5 valid spikes."""
        ts = self.get_valid_spikes(mu_id)
        if len(ts) < 5: 
            return np.nan, np.nan
        
        # Calculate indices based on torque sampling rate (fs_t)
        rec_indices = (ts[:5] * self.fs_t).astype(int)
        derec_indices = (ts[-5:] * self.fs_t).astype(int)
        
        # Mean torque values (clipping ensures we stay within data bounds)
        rec_t = np.mean(self.torque[np.clip(rec_indices, 0, len(self.torque)-1)])
        derec_t = np.mean(self.torque[np.clip(derec_indices, 0, len(self.torque)-1)])
        
        return rec_t, derec_t

    def get_toolbox_metrics(self, mu_id):
        """Uses motor_unit_toolbox to calculate CV."""
        try:
            import motor_unit_toolbox.props as props
            import motor_unit_toolbox.utils as utils
            ts = self.get_valid_spikes(mu_id)
            if len(ts) < 5: return np.nan
            
            # Prepare data for toolbox (needs binary spike train)
            duration_samples = int(np.max(ts) * self.fs_n) + 100
            binary_train = utils.firings_to_binary([(ts * self.fs_n).astype(int)], duration_samples)
            timestamps_vec = np.arange(duration_samples) / self.fs_n
            
            # toolbox returns CoV scaled by 100
            cv_array = props.get_coefficient_of_variation(binary_train, timestamps_vec)
            return cv_array[0]
        except:
            # Fallback if toolbox is not connected
            ts = self.get_valid_spikes(mu_id)
            isis = np.diff(ts)
            return (np.std(isis) / np.mean(isis)) * 100 if len(isis) > 0 else np.nan



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MU Gold Standard Pipeline v28.0 (Multi-rate Fixed)")
        self.resize(1600, 1000)
        self.dm = DataManager()
        self.regions = []

        main_widget = QWidget(); self.setCentralWidget(main_widget)
        self.layout = QVBoxLayout(main_widget)
        self.tabs = QTabWidget(); self.layout.addWidget(self.tabs)

        self.setup_tab = QWidget(); self.decomp_tab = QWidget()
        self.timing_tab = QWidget(); self.audit_tab = QWidget()
        self.tabs.addTab(self.setup_tab, "1. Setup")
        self.tabs.addTab(self.decomp_tab, "2. Decomposition")
        self.tabs.addTab(self.timing_tab, "3. Timing")
        self.tabs.addTab(self.audit_tab, "4. Audit & Export")

        self.init_setup_tab(); self.init_decomp_tab(); self.init_timing_tab(); self.init_audit_tab()

    def init_decomp_tab(self):
        layout = QHBoxLayout(self.decomp_tab)
        ctrls = QVBoxLayout()
        ctrls.addWidget(QLabel("<b>Separation Lab</b>"))
        self.combo_split = QComboBox(); ctrls.addWidget(self.combo_split)
        btn_analyse = QPushButton("Analyse Amplitude"); btn_analyse.clicked.connect(self.on_analyse_amplitude_split); ctrls.addWidget(btn_analyse)
        self.edit_thr = QLineEdit(); self.edit_thr.setPlaceholderText("Threshold (mV)"); ctrls.addWidget(self.edit_thr)
        btn_split = QPushButton("Execute Split"); btn_split.clicked.connect(self.on_split_mu); ctrls.addWidget(btn_split)
        
        ctrls.addWidget(QLabel("\n<b>Merge Lab</b>"))
        self.combo_m1 = QComboBox(); self.combo_m2 = QComboBox()
        ctrls.addWidget(QLabel("Keep MU:")); ctrls.addWidget(self.combo_m1)
        ctrls.addWidget(QLabel("Merge MU:")); ctrls.addWidget(self.combo_m2)
        btn_comp = QPushButton("Compare MUs"); btn_comp.clicked.connect(self.on_compare_mus); ctrls.addWidget(btn_comp)
        
        # Fixed UI: Increased minimum height and enabled word wrap to prevent truncation
        self.lbl_merge = QLabel("Metrics: --")
        self.lbl_merge.setWordWrap(True)
        self.lbl_merge.setMinimumHeight(100) 
        self.lbl_merge.setAlignment(Qt.AlignmentFlag.AlignTop)
        ctrls.addWidget(self.lbl_merge)
        
        btn_merge = QPushButton("Execute Merge"); btn_merge.clicked.connect(self.on_merge_mu); ctrls.addWidget(btn_merge)
        ctrls.addStretch()
        
        plots = QVBoxLayout()
        self.p_split_hist = pg.PlotWidget(title="PTP Distribution"); self.p_split_scatter = pg.PlotWidget(title="PTP vs. Time")
        self.p_merge_temp = pg.PlotWidget(title="Template Overlay"); self.p_merge_raster = pg.PlotWidget(title="Timing Interleave")
        self.p_merge_raster.setYRange(-0.5, 1.5)
        plots.addWidget(self.p_split_hist); plots.addWidget(self.p_split_scatter); plots.addWidget(self.p_merge_temp); plots.addWidget(self.p_merge_raster)
        layout.addLayout(ctrls, 1); layout.addLayout(plots, 4)

    def on_compare_mus(self):
        if not self.sync_mask():
            QMessageBox.warning(self, "Error", "Define analysis windows on the Setup tab first.")
            return
        id_a, id_b = self.combo_m1.currentText(), self.combo_m2.currentText()
        if not id_a or not id_b or id_a == id_b: return
        
        t_a = self.dm.get_template(id_a)
        t_b = self.dm.get_template(id_b)
        if t_a is None or t_b is None: return

        # Shape Metrics
        corr = np.corrcoef(t_a, t_b)[0,1]
        nmse_str = "N/A"
        try:
            import motor_unit_toolbox.muap_comp as muap
            nmse_str = f"{muap.nmse(t_a, t_b):.4f}"
        except:
            mse = np.mean((t_a - t_b)**2)
            energy = np.mean(t_a**2 + t_b**2)
            nmse_str = f"{(mse/energy):.4f} (calc)"

        # Timing: 20ms Coincidence
        ts_a = self.dm.get_valid_spikes(id_a)
        ts_b = self.dm.get_valid_spikes(id_b)
        
        # Calculate Coincidences
        combined = sorted([(t, 'A') for t in ts_a] + [(t, 'B') for t in ts_b])
        coincidences = 0
        for i in range(len(combined)-1):
            if combined[i][1] != combined[i+1][1]:
                if abs(combined[i+1][0] - combined[i][0]) < 0.020:
                    coincidences += 1
        
        # Calculate Percentage relative to the smaller train
        min_spikes = min(len(ts_a), len(ts_b))
        coinc_pct = (coincidences / min_spikes * 100) if min_spikes > 0 else 0
        
        self.lbl_merge.setText(
            f"<b>Corr:</b> {corr:.3f}<br>"
            f"<b>NMSE:</b> {nmse_str}<br>"
            f"<b>Coincidences (&lt;20ms):</b> {coincidences}<br>"
            f"<b>% of smaller MU:</b> {coinc_pct:.1f}%"
        )
        
        self.p_merge_temp.clear(); self.p_merge_temp.plot(t_a, pen='y'); self.p_merge_temp.plot(t_b, pen='c')
        self.p_merge_raster.clear()
        self.p_merge_raster.plot(ts_a, np.ones_like(ts_a), pen=None, symbol='o', symbolBrush='y')
        self.p_merge_raster.plot(ts_b, np.zeros_like(ts_b), pen=None, symbol='o', symbolBrush='c')

    # --- HANDLERS (REMAINDER PRESERVED FROM V27) ---
    def sync_mask(self):
        if not self.regions: return False
        self.dm.update_mask([r.getRegion() for r in self.regions])
        return True

    def on_analyse_amplitude_split(self):
        # 1. Ensure sync_mask() is updated to use fs_t
        if not self.sync_mask():
            QMessageBox.warning(self, "Error", "Define analysis windows on the Setup tab first.")
            return
        
        mu_id = self.combo_split.currentText()
        ts = self.dm.get_valid_spikes(mu_id)
        if len(ts) == 0: 
            print("No valid spikes found for this MU.")
            return
            
        # 2. Ensure PTP calculation uses the correct needle rate (fs_n)
        ptps = self.dm.calculate_ptp_for_timestamps(ts)
        
        self.p_split_hist.clear()
        y, x = np.histogram(ptps, bins=30)
        self.p_split_hist.plot(x, y, stepMode="center", fillLevel=0, brush='b')
        
        self.p_split_scatter.clear()
        self.p_split_scatter.plot(ts, ptps, pen=None, symbol='o', symbolBrush='w')

    def on_split_mu(self):
        mu_id = self.combo_split.currentText(); self.sync_mask()
        try: thr = float(self.edit_thr.text())
        except: return
        ts = self.dm.get_valid_spikes(mu_id); ptps = self.dm.calculate_ptp_for_timestamps(ts)
        self.dm.split_mu(mu_id, thr, ts, ptps); self.update_mu_lists()

    def on_merge_mu(self):
        self.dm.merge_mu(self.combo_m1.currentText(), self.combo_m2.currentText()); self.update_mu_lists()

    def update_mu_lists(self):
        if self.dm.spike_df is None: return
        ids = sorted(self.dm.spike_df[self.dm.spike_df['status']=='Valid']['mu_id'].unique())
        for c in [self.combo_split, self.combo_m1, self.combo_m2, self.combo_time]:
            c.clear(); c.addItems(ids)

    def init_setup_tab(self):
        layout = QHBoxLayout(self.setup_tab); ctrls = QVBoxLayout()
        self.led_t, l1 = self.create_led("Toolbox"); self.led_m, l2 = self.create_led("MAT"); self.led_a, l3 = self.create_led("ANN")
        ctrls.addLayout(l1); ctrls.addLayout(l2); ctrls.addLayout(l3)
        btn_t = QPushButton("Connect Toolbox"); btn_t.clicked.connect(self.on_set_toolbox); ctrls.addWidget(btn_t)
        btn_m = QPushButton("Load .mat"); btn_m.clicked.connect(self.on_load_mat); ctrls.addWidget(btn_m)
        btn_a = QPushButton("Load .ann"); btn_a.clicked.connect(self.on_load_ann); ctrls.addWidget(btn_a)
        self.c_n = QComboBox(); self.c_s = QComboBox(); self.c_t = QComboBox()
        ctrls.addWidget(QLabel("Needle:")); ctrls.addWidget(self.c_n)
        ctrls.addWidget(QLabel("Surface:")); ctrls.addWidget(self.c_s)
        ctrls.addWidget(QLabel("Torque:")); ctrls.addWidget(self.c_t)
        btn_map = QPushButton("Apply Mapping"); btn_map.clicked.connect(self.on_apply_map); ctrls.addWidget(btn_map)
        btn_win = QPushButton("+ Add Analysis Window"); btn_win.clicked.connect(self.on_add_window); ctrls.addWidget(btn_win)
        btn_clear = QPushButton("Clear Windows"); btn_clear.clicked.connect(self.on_clear_windows); ctrls.addWidget(btn_clear)
        ctrls.addStretch()
        self.p_n = pg.PlotWidget(title="Needle EMG"); self.p_s = pg.PlotWidget(title="Surface EMG"); self.p_t = pg.PlotWidget(title="Torque")
        self.p_s.setXLink(self.p_n); self.p_t.setXLink(self.p_n)
        v = QVBoxLayout(); v.addWidget(self.p_n); v.addWidget(self.p_s); v.addWidget(self.p_t)
        layout.addLayout(ctrls, 1); layout.addLayout(v, 4)

    def on_set_toolbox(self):
        p = QFileDialog.getExistingDirectory(self, "Select Toolbox Folder")
        if p:
            sys.path.append(p); self.dm.toolbox_path = p
            try: import motor_unit_toolbox.muap_comp as m; self.update_led(self.led_t, True)
            except: self.update_led(self.led_t, False)

    def on_load_mat(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load .mat", "", "MAT (*.mat)")
        if p and self.dm.load_mat(p):
            self.update_led(self.led_m, True)
            for c in [self.c_n, self.c_s, self.c_t]: c.clear(); c.addItems(self.dm.titles)

    def on_load_ann(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load .ann", "", "ANN (*.ann *.txt *.csv)")
        if p and self.dm.load_ann(p):
            self.update_led(self.led_a, True); self.update_mu_lists()

    def on_apply_map(self):
        if self.dm.raw_data is None: return
        self.dm.map_channels(self.c_n.currentIndex(), self.c_s.currentIndex(), self.c_t.currentIndex())
        # Plotting using individual time vectors
        self.p_n.clear(); self.p_n.plot(np.arange(len(self.dm.needle_emg))/self.dm.fs_n, self.dm.needle_emg, pen='y')
        self.p_s.clear(); self.p_s.plot(np.arange(len(self.dm.surface_emg))/self.dm.fs_s, self.dm.surface_emg, pen='c')
        self.p_t.clear(); self.p_t.plot(np.arange(len(self.dm.torque))/self.dm.fs_t, self.dm.torque, pen='g')

    def init_timing_tab(self):
        layout = QHBoxLayout(self.timing_tab)
        ctrls = QVBoxLayout()
        
        ctrls.addWidget(QLabel("<b>ISI Review Filters</b>"))
        self.combo_time = QComboBox()
        ctrls.addWidget(self.combo_time)
        
        # Threshold input
        h_range = QHBoxLayout()
        self.edit_isi_thr = QLineEdit("20")
        h_range.addWidget(QLabel("Flag ISIs <")); h_range.addWidget(self.edit_isi_thr); h_range.addWidget(QLabel("ms"))
        ctrls.addLayout(h_range)
        
        btn_check = QPushButton("Analyse Timing")
        btn_check.clicked.connect(self.on_analyse_isi)
        ctrls.addWidget(btn_check)
        
        ctrls.addWidget(QLabel("\n<b>Review Outliers</b>"))
        self.table_viol = QTableWidget(0, 3)
        self.table_viol.setHorizontalHeaderLabels(["ISI (ms)", "T1 (s)", "T2 (s)"])
        self.table_viol.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_viol.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table_viol.itemSelectionChanged.connect(self.on_violation_selected)
        ctrls.addWidget(self.table_viol)
        
        # Specific Delete Buttons
        h_del = QHBoxLayout()
        self.btn_del1 = QPushButton("Del Spike 1"); self.btn_del1.clicked.connect(lambda: self.delete_specific(1))
        self.btn_del2 = QPushButton("Del Spike 2"); self.btn_del2.clicked.connect(lambda: self.delete_specific(2))
        h_del.addWidget(self.btn_del1); h_del.addWidget(self.btn_del2)
        ctrls.addLayout(h_del)
        
        self.btn_bulk_del = QPushButton("BULK DELETE Selected (Spike 2)")
        self.btn_bulk_del.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold;")
        self.btn_bulk_del.clicked.connect(self.on_bulk_delete)
        ctrls.addWidget(self.btn_bulk_del)
        
        ctrls.addStretch()
        
        plots = QVBoxLayout()
        self.p_isi_hist = pg.PlotWidget(title="ISI Histogram (ms) - Drag Yellow Line")
        self.p_isi_hist.setLabel('bottom', 'ISI (ms)')
        self.p_isi_hist.getAxis('bottom').enableAutoSIPrefix(False)

        self.isi_thr_line = pg.InfiniteLine(pos=20, angle=90, movable=True, pen=pg.mkPen('y', width=2))
        self.p_isi_hist.addItem(self.isi_thr_line)
        # Link the drag of the line to the box value
        self.isi_thr_line.sigPositionChanged.connect(lambda: self.edit_isi_thr.setText(f"{self.isi_thr_line.value():.1f}"))
        
        self.p_timing_zoom = pg.PlotWidget(title="Needle Zoom (S1=Purple, S2=Blue)")
        
        self.p_wave_station = pg.PlotWidget(title="Waveform Overlay (White=Mean, Purple=S1, Blue=S2)")
        self.p_wave_station.setLabel('bottom', 'Time (ms)')
        self.lbl_wave_metrics = QLabel("NMSE S1: -- | NMSE S2: --")
        self.lbl_wave_metrics.setStyleSheet("font-weight: bold; color: yellow; background-color: #2c3e50; padding: 5px;")
        
        plots.addWidget(self.p_isi_hist)
        plots.addWidget(self.p_timing_zoom)
        plots.addWidget(self.p_wave_station)
        plots.addWidget(self.lbl_wave_metrics)
        
        layout.addLayout(ctrls, 1)
        layout.addLayout(plots, 4)



    def on_analyse_isi(self):
        self.sync_mask()
        mu_id = self.combo_time.currentText()
        ts = self.dm.get_valid_spikes(mu_id)
        if len(ts) < 2: return
        
        # 1. Calculate ISIs and convert to ms
        isis_ms = np.diff(ts) * 1000
        
        # --- NEW: Filter data to 200ms limit ---
        plot_data = isis_ms[isis_ms <= 200]
        
        self.p_isi_hist.clear()
        self.p_isi_hist.addItem(self.isi_thr_line)
        
        # Calculate histogram with a fixed range of 0 to 200
        y, x = np.histogram(plot_data, bins=50, range=(0, 200))
        self.p_isi_hist.plot(x, y, stepMode="center", fillLevel=0, brush=(200, 50, 50, 100))
        
        # Force the plot widget to stay within 0-200ms
        self.p_isi_hist.setXRange(0, 200, padding=0)
        # ---------------------------------------
        
        try: thr = float(self.edit_isi_thr.text())
        except: thr = 20.0
        self.isi_thr_line.setValue(thr)


        self.table_viol.setRowCount(0)
        self.current_violations = []
        
        review_list = []
        for i in range(len(ts)-1):
            isi_val = (ts[i+1] - ts[i]) * 1000
            if isi_val < thr:
                review_list.append({'isi': isi_val, 't1': ts[i], 't2': ts[i+1]})
        
        # Sort Lowest to Highest
        review_list = sorted(review_list, key=lambda x: x['isi'])
        self.current_violations = review_list
        
        for v in review_list:
            r = self.table_viol.rowCount()
            self.table_viol.insertRow(r)
            self.table_viol.setItem(r, 0, QTableWidgetItem(f"{v['isi']:.1f}"))
            self.table_viol.setItem(r, 1, QTableWidgetItem(f"{v['t1']:.4f}"))
            self.table_viol.setItem(r, 2, QTableWidgetItem(f"{v['t2']:.4f}"))

    def on_violation_selected(self):
        sel = self.table_viol.selectedItems()
        if not sel: return
        row = sel[0].row() # Use first selected row for plotting
        v = self.current_violations[row]
        mu_id = self.combo_time.currentText()
        
        # Plot 2: Zoom
        self.p_timing_zoom.setXRange(v['t1']-0.02, v['t2']+0.02)
        self.p_timing_zoom.clear()
        t_ax = np.arange(len(self.dm.needle_emg))/self.dm.fs_n
        self.p_timing_zoom.plot(t_ax, self.dm.needle_emg, pen='y')
        self.p_timing_zoom.addItem(pg.InfiniteLine(pos=v['t1'], pen=pg.mkPen('#9b59b6', width=2)))
        self.p_timing_zoom.addItem(pg.InfiniteLine(pos=v['t2'], pen=pg.mkPen('#3498db', width=2)))

        # Plot 3: Waveform Overlay
        mean_t = self.dm.get_template(mu_id)
        wf1 = self.dm.get_spike_segment(v['t1'])
        wf2 = self.dm.get_spike_segment(v['t2'])
        
        self.p_wave_station.clear()
        if mean_t is not None:
            t_ms = np.linspace(-5, 5, len(mean_t))
            self.p_wave_station.plot(t_ms, mean_t, pen=pg.mkPen('w', width=3))
            nmse1 = self.dm.calculate_nmse(wf1, mean_t) if wf1 is not None else 1.0
            nmse2 = self.dm.calculate_nmse(wf2, mean_t) if wf2 is not None else 1.0
            
            if wf1 is not None: self.p_wave_station.plot(t_ms, wf1, pen=pg.mkPen('#9b59b6', width=1.5))
            if wf2 is not None: self.p_wave_station.plot(t_ms, wf2, pen=pg.mkPen('#3498db', width=1.5))
            self.lbl_wave_metrics.setText(f"NMSE S1: {nmse1:.4f} | NMSE S2: {nmse2:.4f}")

        # Store global indices for targeted deletion
        m1 = self.dm.spike_df[(self.dm.spike_df['mu_id']==mu_id) & np.isclose(self.dm.spike_df['timestamp'], v['t1'])]
        m2 = self.dm.spike_df[(self.dm.spike_df['mu_id']==mu_id) & np.isclose(self.dm.spike_df['timestamp'], v['t2'])]
        self.current_pair_indices = (m1.index[0] if not m1.empty else None, m2.index[0] if not m2.empty else None)

    def delete_specific(self, num):
        if not hasattr(self, 'current_pair_indices'): return
        idx = self.current_pair_indices[0] if num == 1 else self.current_pair_indices[1]
        if idx is not None:
            self.dm.spike_df.at[idx, 'status'] = 'Eliminated'
            self.on_analyse_isi()

    def on_bulk_delete(self):
        selected_rows = list(set(index.row() for index in self.table_viol.selectedIndexes()))
        if not selected_rows: return
        mu_id = self.combo_time.currentText()
        for row in selected_rows:
            v = self.current_violations[row]
            match = self.dm.spike_df[(self.dm.spike_df['mu_id']==mu_id) & np.isclose(self.dm.spike_df['timestamp'], v['t2'])]
            if not match.empty:
                self.dm.spike_df.at[match.index[0], 'status'] = 'Eliminated'
        self.on_analyse_isi()

    def init_audit_tab(self):
        layout = QVBoxLayout(self.audit_tab); self.table_audit = QTableWidget(0, 7)
        self.table_audit.setHorizontalHeaderLabels(["MU ID", "PNR", "CV%", "Recruit", "Derecruit", "Count", "Status"])
        layout.addWidget(self.table_audit)
        btn_refresh = QPushButton("Refresh Health Check"); btn_refresh.clicked.connect(self.on_refresh_audit); layout.addWidget(btn_refresh)
        btn_export = QPushButton("EXPORT MAT FOR TMS APP"); btn_export.clicked.connect(self.on_export)
        btn_export.setStyleSheet("background-color: #2ecc71; height: 50px; font-weight: bold;"); layout.addWidget(btn_export)

    def on_refresh_audit(self):
        """Populates the Audit table with metrics for current valid MUs."""
        if not self.sync_mask():
            QMessageBox.warning(self, "Warning", "Please define Analysis Windows on the Setup tab first.")
            return

        self.table_audit.setRowCount(0)
        
        # Pull IDs only for currently 'Valid' spikes. 
        # This automatically excludes old IDs that were split or merged.
        mu_ids = sorted(self.dm.spike_df[self.dm.spike_df['status'] == 'Valid']['mu_id'].unique())
        
        for mid in mu_ids:
            ts = self.dm.get_valid_spikes(mid)
            if len(ts) < 2: 
                continue
            
            # 1. Calculate Metrics
            pnr = self.dm.calculate_pnr(mid)
            cv = self.dm.get_toolbox_metrics(mid)
            rec, derec = self.dm.get_recruitment_info(mid)
            
            # 2. Create Row
            row = self.table_audit.rowCount()
            self.table_audit.insertRow(row)
            
            # 3. Populate Columns
            self.table_audit.setItem(row, 0, QTableWidgetItem(str(mid)))
            self.table_audit.setItem(row, 1, QTableWidgetItem(f"{pnr:.1f}"))
            self.table_audit.setItem(row, 2, QTableWidgetItem(f"{cv:.1f}"))
            self.table_audit.setItem(row, 3, QTableWidgetItem(f"{rec:.3f}" if not np.isnan(rec) else "N/A"))
            self.table_audit.setItem(row, 4, QTableWidgetItem(f"{derec:.3f}" if not np.isnan(derec) else "N/A"))
            self.table_audit.setItem(row, 5, QTableWidgetItem(str(len(ts))))
            
            # 4. Status Logic (GOLD if PNR > 30 and CV < 20%)
            status_text = "GOLD" if (pnr > 30 and cv < 20) else "REVIEW"
            status_item = QTableWidgetItem(status_text)
            if status_text == "GOLD":
                status_item.setBackground(Qt.GlobalColor.green)
            elif cv > 40 or pnr < 15:
                status_item.setBackground(Qt.GlobalColor.red)
            
            self.table_audit.setItem(row, 6, status_item)

    def on_export(self):
        self.sync_mask(); path, _ = QFileDialog.getSaveFileName(self, "Export MAT", "", "MAT (*.mat)")
        if not path: return
        mu_ids = sorted(self.dm.spike_df[self.dm.spike_df['status']=='Valid']['mu_id'].unique())
        export_data = {mid: {"spikes": self.dm.get_valid_spikes(mid)} for mid in mu_ids}
        scipy.io.savemat(path, {"mu_export": export_data})

    def on_add_window(self):
        if self.dm.torque is None: return
        r = pg.LinearRegionItem([0, 5]); r.setBrush(pg.mkBrush(255,0,0,50)); self.p_t.addItem(r); self.regions.append(r)

    def on_clear_windows(self):
        for r in self.regions: self.p_t.removeItem(r)
        self.regions = []

    def create_led(self, text):
        l = QHBoxLayout(); led = QLabel(); led.setFixedSize(14,14)
        led.setStyleSheet("background-color: #7f8c8d; border-radius: 7px;")
        l.addWidget(led); l.addWidget(QLabel(text))
        return led, l

    def update_led(self, led, active):
        color = "#2ecc71" if active else "#e74c3c"
        led.setStyleSheet(f"background-color: {color}; border-radius: 7px; border: 1px solid black;")

if __name__ == "__main__":
    app = QApplication(sys.argv); w = MainWindow(); w.show(); sys.exit(app.exec())