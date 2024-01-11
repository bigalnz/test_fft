
import tkinter as tk

from tkinter import ttk
from tkinter import filedialog
from tkinter.ttk import *

import argparse
import asyncio
import json
from kiwitracker.common import SamplesT, SampleConfig, ProcessConfig
from kiwitracker.sample_reader import run_main, run_readonly, run_from_disk
import sys

def load_default_settings():
    try:
        with open('default_settings.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}  # Return an empty dictionary if file is not found

def save_default_settings(defaults):
    try:
        with open('default_settings.json', 'w') as file:
            return json.dump(defaults, file)
    except FileNotFoundError:
        print("Failed to save Defaults", e)

# Modified main method to accept arguments directly
def main_gui(args):
    sample_config = SampleConfig(
        sample_rate=args['sample_rate'], center_freq=args['center_freq'],
        gain=args['gain'], bias_tee_enable=args['bias_tee'], read_size=args['chunk_size'],
    )
    process_config = ProcessConfig(
        sample_config=sample_config, carrier_freq=args['carrier'],
    )

    if args['infile']:
        run_from_disk(
            process_config=process_config,
            filename=args['infile'],
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



class TextRedirector(object):
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        self.widget.insert(tk.END, str)
        self.widget.see(tk.END)

    def flush(self):
        pass


# Tkinter GUI setup
def run_gui():
    window = tk.Tk()
    window.title("SDR Sample Processor")

    def select_file(entry):
        filename = filedialog.askopenfilename()
        entry.delete(0, tk.END)
        entry.insert(0, filename)

    def on_run_click():
        args = {
            'infile': infile_entry.get(),
            'outfile': outfile_entry.get(),
            'max_samples': int(max_samples_entry.get()) if max_samples_entry.get() else None,
            'chunk_size': int(chunk_size_entry.get()),
            'sample_rate': float(sample_rate_entry.get()),
            'center_freq': float(center_freq_entry.get()),
            'gain': float(gain_entry.get()),
            'bias_tee': bias_tee_var.get(),
            'carrier': float(carrier_entry.get())
        }
        sys.stdout = TextRedirector(output_text)
        save_default_settings(args)
        main_gui(args)

    # Create input fields
    infile_label = ttk.Label(window, text="Input File:")
    infile_entry = ttk.Entry(window)
    infile_button = ttk.Button(window, text="Select File", command=lambda: select_file(infile_entry))

    outfile_label = ttk.Label(window, text="Output File:")
    outfile_entry = ttk.Entry(window)
    outfile_button = ttk.Button(window, text="Select File", command=lambda: select_file(outfile_entry))

    max_samples_label = ttk.Label(window, text="Max Samples:")
    max_samples_entry = ttk.Entry(window)

    chunk_size_label = ttk.Label(window, text="Chunk Size:")
    chunk_size_entry = ttk.Entry(window)

    sample_rate_label = ttk.Label(window, text="Sample Rate:")
    sample_rate_entry = ttk.Entry(window)

    center_freq_label = ttk.Label(window, text="Center Frequency:")
    center_freq_entry = ttk.Entry(window)

    gain_label = ttk.Label(window, text="Gain:")
    gain_entry = ttk.Entry(window)

    bias_tee_label = ttk.Label(window, text="Bias Tee:")
    bias_tee_var = tk.BooleanVar()
    bias_tee_check = ttk.Checkbutton(window, variable=bias_tee_var)

    carrier_label = ttk.Label(window, text="Carrier Frequency:")
    carrier_entry = ttk.Entry(window)
    carrier_entry = ttk.Entry(window)

    run_button = ttk.Button(window, text="Run", command=on_run_click)

    pad = ttk.Label(window, text="")

    defaults = load_default_settings()
    infile_entry.insert(0, defaults.get('infile', ''))
    outfile_entry.insert(0, defaults.get('outfile', ''))
    max_samples_entry.insert(0, defaults.get('max_samples', ''))
    chunk_size_entry.insert(0, defaults.get('chunk_size', ''))
    sample_rate_entry.insert(0, defaults.get('sample_rate', ''))
    center_freq_entry.insert(0, defaults.get('center_freq', ''))
    gain_entry.insert(0, defaults.get('gain', ''))
    bias_tee_var.set(defaults.get('bias_tee', False))
    carrier_entry.insert(0, defaults.get('carrier', ''))

    # Arrange widgets using grid
    labels = [infile_label, outfile_label, max_samples_label, chunk_size_label,
              sample_rate_label, center_freq_label, gain_label, carrier_label, bias_tee_label]
    entries = [infile_entry, outfile_entry, max_samples_entry, chunk_size_entry,
               sample_rate_entry, center_freq_entry, gain_entry, carrier_entry, bias_tee_check]
    buttons = [infile_button, outfile_button, pad, pad, pad, pad, pad, pad, run_button]

    output_text = tk.Text(window, height=10)


    # Place items into grid
    output_text.grid(row=len(labels) + len(buttons), column=0, columnspan=3)
    
    for i, label in enumerate(labels):
        label.grid(row=i, column=0, sticky='w', padx=10, pady=5)

    for i, entry in enumerate(entries):
        entry.grid(row=i, column=1, padx=5)
        if entry is not bias_tee_check:
            entry.insert(0, defaults.get(entry._name, ''))  # Use entry widget name to get default value

    for i, button in enumerate(buttons):
        button.grid(row=i, column=2)

    window.columnconfigure(0, weight=1)
    window.columnconfigure(1, weight=3)
    window.columnconfigure(2, weight=1)

    window.mainloop()

if __name__ == "__main__":
    run_gui()
