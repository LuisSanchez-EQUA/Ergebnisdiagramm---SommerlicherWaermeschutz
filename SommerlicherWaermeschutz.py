"""
Dieses Skript analysiert IDA ICE-Simulationsdaten und erstellt automatisch Diagramme_SommerlicherWärmeschutz für die Zone mit der höchsten Lufttemperatur innerhalb einer Woche. 
Der Benutzer wird aufgefordert, Modell- und Zonennamen für die Diagramme auszuwählen. 
Außerdem muss der Benutzer die Ausgabedatei auswählen, die von IDA ICE erzeugt wurden.
Die generierten Diagramme werden strukturiert in einem Ordner gespeichert: Diagramme > Modellname > Zonenname.

2025-08-12, LS
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.dates import DateFormatter
from datetime import timedelta
import tkinter as tk
from tkinter import Tk, filedialog, simpledialog


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(ROOT_DIR, "Diagramme_SommerlicherWärmeschutzC1")
print(f"Plots will be saved to: {PLOTS_DIR}")

EXPECTED_PATTERNS = [
    "local-ch-comfort-sia-180",
    #"temperatures",
    #"heat_balance"
]

import tkinter as tk
from tkinter import simpledialog

class ModelZoneDialog(simpledialog.Dialog):
    def __init__(self, parent, default_model="Model_1", default_zone="Zone_1", default_mech_vent=False):
        self.default_model = default_model
        self.default_zone = default_zone
        self.default_mech_vent = default_mech_vent
        super().__init__(parent, title="Enter Model and Zone Names")

    def body(self, master):
        tk.Label(master, text="Gebäude/Modell:").grid(row=0, column=0, sticky="e", padx=10, pady=5)
        tk.Label(master, text="Raum/Zone:").grid(row=1, column=0, sticky="e", padx=10, pady=5)

        self.model_entry = tk.Entry(master, width=30)
        self.zone_entry = tk.Entry(master, width=30)
        self.model_entry.insert(0, self.default_model)
        self.zone_entry.insert(0, self.default_zone)

        self.model_entry.grid(row=0, column=1, padx=10, pady=5)
        self.zone_entry.grid(row=1, column=1, padx=10, pady=5)

        # Checkbox for mechanical ventilation
        self.mech_vent_var = tk.BooleanVar(value=self.default_mech_vent)
        self.mech_vent_check = tk.Checkbutton(
            master, 
            text="Mechanische Lüftung", 
            variable=self.mech_vent_var
        )
        self.mech_vent_check.grid(row=2, column=1, columnspan=2, padx=10, pady=5, sticky="w")

        return self.model_entry

    def apply(self):
        self.model_name = self.model_entry.get().strip()
        self.zone_name = self.zone_entry.get().strip()
        self.has_mech_vent = self.mech_vent_var.get()  # True/False

def ask_model_and_zone_names():
    root = tk.Tk()
    root.withdraw()
    dialog = ModelZoneDialog(root)

    model = getattr(dialog, 'model_name', None)
    zone = getattr(dialog, 'zone_name', None)
    mech_vent = getattr(dialog, 'has_mech_vent', None)

    if not model or not zone:
        raise ValueError("Both model and zone names are required.")

    return model, zone, mech_vent


def select_zone_files():
    #print("Please select the 3 .prn files for a single zone (comfort, temperature, heat balance).")

    root = Tk()
    root.withdraw()
    selected_files = filedialog.askopenfilenames(
        title="Select LOCAL-CH-COMFORT-SIA-180.prn file for one zone",
        filetypes=[("PRN files", "*.prn")],
    )

    if len(selected_files) != 1:
        raise ValueError("You must select only one file.")

    zone_prn_paths = {}
    for file_path in selected_files:
        file_name = os.path.basename(file_path)
        file_name_lower = file_name.lower()

        matched = False
        for pattern in EXPECTED_PATTERNS:
            if pattern in file_name_lower:
                zone_prn_paths[pattern] = file_path
                matched = True
                break

        if not matched:
            raise ValueError(f"Unexpected file selected: {file_name}")

    print(f"Selected files: {list(zone_prn_paths.keys())}")
    #print(f"Paths: {list(zone_prn_paths.values())}")
    return zone_prn_paths

def preprocess_headers(df):
    """Fixes headers of the DataFrame."""
    try:
        columns = df.columns.tolist()[1:]
        df = df.drop(columns=df.columns[-1])
        df.columns = columns
        return df
    except IndexError:
        print(f"Warning: Could not preprocess headers for DataFrame. Check file format.")
        return df 

def IDAICE_toTS(x, t):
    return np.datetime64(t, 'm') + int(x * 60)

def format_change(df, year=2025):
    time_0 = f"{year}-01-01 01:00:00" 
    if df.shape[1] > 1:
        
        df.iloc[:, 1] = df.iloc[:, 0].apply(lambda x: IDAICE_toTS(x, time_0))
        df = df.rename(columns={df.columns[1]: 'Date'})
        df.set_index("Date", inplace=True)
    else:
       
        print(f"Warning: DataFrame has unexpected shape for format_change. Columns: {df.columns.tolist()}")
        
        if 'time' in df.columns:
            df.set_index('time', inplace=True)
        else:
            df.index = pd.date_range(start=time_0, periods=len(df), freq='H') 


    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


# _____________________________________________________________________________
# PLOTTING FUNCTIONS
# _____________________________________________________________________________

def save_plot(fig, model_name, zone_name, file_base_name):
    """Saves the generated plot to the designated model/zone subdirectory."""
    model_plot_dir = os.path.join(PLOTS_DIR, model_name)
    zone_plot_dir = os.path.join(model_plot_dir, zone_name)
    
    
    os.makedirs(zone_plot_dir, exist_ok=True) 

    plot_path = os.path.join(zone_plot_dir, f"{model_name}_{zone_name}_{file_base_name}.png")
    try:
        fig.savefig(plot_path)
        print(f"Plot saved: {file_base_name} in {plot_path}")
    except Exception as e:
        print(f"Error saving plot {file_base_name}: {e}")
    finally:
        plt.close(fig) 


# Helper function for comfort
def plot_crossdependency(ax, x_data, y_data, label, color, s=5):
    """Helper function to plot cross-dependency."""
    ax.scatter(x_data, y_data, label=label, color=color, s=s)

def plot_comfort_sia180(data, model_name, zone_name, mech_vent):
    """Plots comfort data based on SIA 180 standard."""
    if data.empty:
        print(f"Skipping comfort plot for {model_name} - {zone_name}: No data.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    target_column = 'tairma'
    target_data = data[target_column]
    #print(f"Data head:\n{data.head()}")

    limit_col = 'togactive' if mech_vent else 'togpassiv'
    if limit_col in data.columns:
        data_hourly = data.resample('H').mean()

        exceed_mask = data_hourly['toper'] > data_hourly[limit_col]
        exceed_hours = exceed_mask.sum()  
        u_exceed_mask = data_hourly['toper'] > data_hourly['tug']
        u_exceed_hours = u_exceed_mask.sum()

        print(f"{model_name} - {zone_name}: {exceed_hours} hours above {limit_col}")
        print(f"{model_name} - {zone_name}: {u_exceed_hours} hours below 'Untere Grenze'")

        # Optional: add text on plot
        ax.text(
            1.4, 0.02,
            f"Übertemperaturgradstunden:\n     {exceed_hours} h\n Untertemperaturgradstunden:\n     {u_exceed_hours} h",
            transform=ax.transAxes,
            fontsize=10,
            ha='right',
            va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    else:
        print(f"{model_name} - {zone_name}: limit column '{limit_col}' not found.")

    # All possible lines
    plot_cols_all = ['togactive', 'togpassiv', 'toper', 'tug']
    labels_all = [
        "Obere Grenze\n (aktiver-Kühlung)",
        "Obere Grenze\n (passiver-Kühlung)",
        "Operative Temperatur",
        "Untere Grenze"
    ]
    colors_all = ['green', 'blue', 'chocolate', 'magenta']

    # Filter based on mech_vent flag
    filtered_items = [
        (col, label, color)
        for col, label, color in zip(plot_cols_all, labels_all, colors_all)
        if not (mech_vent and col == 'togpassiv') and not (not mech_vent and col == 'togactive')
    ]

    # Reorder so 'toper' is plotted first
    filtered_items.sort(key=lambda x: 0 if x[0] == 'toper' else 1)

    # Plot each line (toper first visually)
    for col, label, color in filtered_items:
        if col in data.columns and col != target_column:
            plot_crossdependency(ax, target_data, data[col], label, color)

    # Keep legend in the original logical order
    legend_labels = [label for col, label, color in zip(plot_cols_all, labels_all, colors_all)
                     if (col, label, color) in filtered_items]
    legend_colors = [color for col, label, color in zip(plot_cols_all, labels_all, colors_all)
                     if (col, label, color) in filtered_items]
    handles = [plt.Line2D([0], [0], color=c, lw=2) for c in legend_colors]
    ax.legend(handles, legend_labels, loc='upper left', bbox_to_anchor=(1.1, 0.8), labelspacing=1.5, frameon=False)

    ax.set_xlabel("Gleitendes Mittel Aussenlufttemperatur über 48 Std", fontsize=12)
    ax.set_ylabel("Temperatur (°C)", fontsize=12)

    file_base_name = "LOCAL-CH-COMFORT-SIA-180"
    ax.set_title(f"{model_name} - {zone_name} - \n{file_base_name}", fontsize=10)
    ax.grid(True, zorder=0)
   
    fig.subplots_adjust(right=0.7, top=0.9)

    save_plot(fig, model_name, zone_name, file_base_name)



# _____________________________________________________________________________
# CORE LOGIC
# _____________________________________________________________________________

def create_zone_plot_directories(model_name, zone_name):
    model_plot_dir = os.path.join(PLOTS_DIR, model_name)
    zone_plot_dir = os.path.join(model_plot_dir, zone_name)
    os.makedirs(zone_plot_dir, exist_ok=True)


def process_zone_data(model_name, zone_name, zone_prn_paths, mech_vent):
    print(f"\nProcessing Model: '{model_name}', Zone: '{zone_name}'")
    create_zone_plot_directories(model_name, zone_name)


    comfort_data = None

    if "local-ch-comfort-sia-180" in zone_prn_paths:
        try:
            path = zone_prn_paths["local-ch-comfort-sia-180"]
            df = pd.read_csv(path, delim_whitespace=True)

            
            rename_map = {
                'tubk_aktiv': 'togactive',
                'tubk_passiv': 'togpassiv',
                'top': 'toper',
                'tul': 'tug'
            }
            df.rename(columns=rename_map, inplace=True)

            comfort_data = preprocess_headers(df)
            comfort_data = format_change(comfort_data)
            comfort_data = comfort_data[~comfort_data.index.duplicated(keep='first')] # Ensure unique index
            print(f"  Loaded LOCAL-CH-COMFORT-SIA-180.prn")
    
  
  
        except Exception as e:
            print(f"  Error loading LOCAL-CH-COMFORT-SIA-180.prn for {zone_name}: {e}")
 
   
    if comfort_data is not None and not comfort_data.empty:
        # Check if critical columns have data before plotting
        required_cols = ['tairma', 'togactive', 'togpassiv', 'toper', 'tug']
        if all(col in comfort_data.columns and not comfort_data[col].isnull().all() for col in required_cols):
            plot_comfort_sia180(comfort_data, model_name, zone_name, mech_vent)
        else:
            missing_cols_info = [col for col in required_cols if col not in comfort_data.columns]
            all_nan_cols_info = [col for col in required_cols if col in comfort_data.columns and comfort_data[col].isnull().all()]
            print(f"Skipping comfort plot for {model_name} - {zone_name}: Missing columns {missing_cols_info} or all NaN in {all_nan_cols_info}.")


    print(f"======================================")


# _____________________________________________________________________________
# MAIN
# _____________________________________________________________________________

def main():
    """Main function to execute the script."""
    os.makedirs(PLOTS_DIR, exist_ok=True) # Ensure the main plots directory exists

    # Prompt user to enter model and zone names
    model_name, zone_name, mech_vent = ask_model_and_zone_names()
    print(f"\nSelected Model: {model_name}, Zone: {zone_name}, Mechanical Ventilation: {mech_vent}")

    # Ask user to select the 3 required PRN files
    try:
        zone_prn_paths = select_zone_files()
        
    except Exception as e:
        print(f"Error selecting files: {e}")
        return

    if not zone_prn_paths:
        print("No PRN files selected.")
        return

    process_zone_data(model_name, zone_name, zone_prn_paths, mech_vent)

    print("\nAll processing complete!")



if __name__ == "__main__":
    main()