import sys
import threading
import asyncio
import json

import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


from kiwitracker.common import SamplesT, SampleConfig, ProcessConfig
from kiwitracker.sample_reader import run_main, run_readonly, run_from_disk

def main_gui(args):
    sample_config = SampleConfig(
        sample_rate=args['sample_rate'], center_freq=args['center_freq'],
        gain=args['gain'], bias_tee_enable=args['bias_tee'], read_size=args['chunk_size'],
    )
    process_config = ProcessConfig(
        sample_config=sample_config, carrier_freq=args['carrier'],
    )
    samples = []
    if args['infile']:
        samples = np.load(args['infile'])[:args['max_samples']]
        plt.plot(samples)
        plt.show()
        run_from_disk(
            process_config=process_config,
            filename=args['infile'],
            samples=samples
        )
        

    elif args['outfile']:
        assert args['max_samples'] is not None
        asyncio.run(
            run_readonly(
                sample_config=sample_config,
                filename=args['outfile'],
                max_samples=args['max_samples'],
            )
        )
    else:
        asyncio.run(
            run_main(sample_config=sample_config, process_config=process_config)
        )




class FileHandler():
    def load_default_settings(self):
        try:
            with open('default_settings.json', 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return {}  # Return an empty dictionary if file is not found

    def save_default_settings(self, defaults):
        try:
            with open('default_settings.json', 'w') as file:
                return json.dump(defaults, file)
        except FileNotFoundError:
            print("Failed to save Defaults", FileNotFoundError)

    def select_file(self, entry):
            filename = filedialog.askopenfilename()
            entry.delete(0, tk.END)
            entry.insert(0, filename)




class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.configure(state="disabled")
    def flush(self):
        pass



class Gui:
    def __init__(self):
        self.window = tk.Tk()
        self.file_handler = FileHandler()
        self.window.title("SDR Sample Processor")
        self.window.configure(background='white')
        self.defaults = self.file_handler.load_default_settings()
        self.setup_widgets()


        self.xdata, self.ydata = [], []

        #self.ani = FuncAnimation(self.fig, self.update_plot, frames=self.data_stream, blit=True)

        num_rows = 9  # 
        num_columns = 3  # 

        for i in range(num_rows):
            self.window.rowconfigure(i, weight=1)  # Makes rows resizable
        
        self.window.columnconfigure(0, weight=0)  # Makes column non-resizable
        self.window.columnconfigure(1, weight=1, minsize=80)  # Makes column resizable
        self.window.columnconfigure(2, weight=0)  # Makes column non-resizable
        self.window.columnconfigure(3, weight=3)  # Makes column resizable


        self.window.mainloop()

    def setup_widgets(self):
        self.labels, self.entries, self.buttons = self.get_elements(self.window, self.defaults)
        
        self.output_text = tk.Text(self.window)
        self.output_text.grid(row=0, column=3, rowspan=10, columnspan=3, sticky="nsew")


        self.scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=self.output_text.yview)
        self.scrollbar.grid(row=0, column=4, rowspan=10, sticky="ns")
        self.output_text.configure(yscrollcommand=self.scrollbar.set)

        for i, label in enumerate(self.labels):
            label.grid(row=i, column=0, sticky='w', padx=10, pady=5)

        for i, entry in enumerate(self.entries):
            entry.grid(row=i, column=1, padx=5, sticky='ew')

        for i, button in enumerate(self.buttons):
            button.grid(row=i, column=2, padx=5)

        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=3)
        self.window.columnconfigure(2, weight=1)

    def on_run_click(self):
        args = {
            'infile': self.infile_entry.get(),
            'outfile': self.outfile_entry.get(),
            'max_samples': int(float(self.max_samples_entry.get())) if self.max_samples_entry.get() else None,
            'chunk_size': int(float(self.chunk_size_entry.get())),
            'sample_rate': float(self.sample_rate_entry.get()),
            'center_freq': float(self.center_freq_entry.get()),
            'gain': float(self.gain_entry.get()),
            'bias_tee': self.bias_tee_var.get(),
            'carrier': float(self.carrier_entry.get())
        }
        sys.stdout = TextRedirector(self.output_text)
        self.file_handler.save_default_settings(args)
        self.task_thread = threading.Thread(target=self.run, args=(args,))
        self.task_thread.start()

    def run(self, args):
        self.run_button['state'] = 'disabled'  # Disable the run button
        try: main_gui(args)
        finally:
            self.run_button['state'] = 'enabled'  # Enable the run button

    def get_elements(self, window, defaults):
        self.infile_label = ttk.Label(window, text="Input File:")
        self.infile_entry = ttk.Entry(window)
        self.infile_button = ttk.Button(window, text="Select File", command=lambda: self.file_handler.select_file(self.infile_entry))

        self.outfile_label = ttk.Label(window, text="Output File:")
        self.outfile_entry = ttk.Entry(window)
        self.outfile_button = ttk.Button(window, text="Select File", command=lambda: self.file_handler.select_file(self.outfile_entry))

        self.max_samples_label = ttk.Label(window, text="Max Samples:")
        self.max_samples_entry = ttk.Entry(window)

        self.chunk_size_label = ttk.Label(window, text="Chunk Size:")
        self.chunk_size_entry = ttk.Entry(window)

        self.sample_rate_label = ttk.Label(window, text="Sample Rate:")
        self.sample_rate_entry = ttk.Entry(window)

        self.center_freq_label = ttk.Label(window, text="Center Frequency:")
        self.center_freq_entry = ttk.Entry(window)

        self.gain_label = ttk.Label(window, text="Gain:")
        self.gain_entry = ttk.Entry(window)

        self.bias_tee_label = ttk.Label(window, text="Bias Tee:")
        self.bias_tee_var = tk.BooleanVar()
        self.bias_tee_check = ttk.Checkbutton(window, variable=self.bias_tee_var)

        self.carrier_label = ttk.Label(window, text="Carrier Frequency:")
        self.carrier_entry = ttk.Entry(window)
        self.carrier_entry = ttk.Entry(window)

        self.run_button = ttk.Button(window, text="Run", command=self.on_run_click)

        pad = ttk.Label(window, text="")
        
        def insert_default_value(entry, default_value):
            if default_value is not None:
                entry.insert(0, default_value)

        insert_default_value(self.infile_entry, defaults.get('infile'))
        insert_default_value(self.outfile_entry, defaults.get('outfile'))
        insert_default_value(self.max_samples_entry, defaults.get('max_samples'))
        insert_default_value(self.chunk_size_entry, defaults.get('chunk_size'))
        insert_default_value(self.sample_rate_entry, defaults.get('sample_rate'))
        insert_default_value(self.center_freq_entry, defaults.get('center_freq'))
        insert_default_value(self.gain_entry, defaults.get('gain'))
        self.bias_tee_var.set(defaults.get('bias_tee', False))
        insert_default_value(self.carrier_entry, defaults.get('carrier'))

        # Arrange widgets using grid
        labels = [self.infile_label, self.outfile_label, self.max_samples_label, self.chunk_size_label,
                self.sample_rate_label, self.center_freq_label, self.gain_label, self.carrier_label, self.bias_tee_label]
        entries = [self.infile_entry, self.outfile_entry, self.max_samples_entry, self.chunk_size_entry,
                self.sample_rate_entry, self.center_freq_entry, self.gain_entry, self.carrier_entry, self.bias_tee_check]
        buttons = [self.infile_button, self.outfile_button, pad, pad, pad, pad, pad, pad, self.run_button]
        return labels, entries, buttons

if __name__ == "__main__":
    gui_app = Gui()
