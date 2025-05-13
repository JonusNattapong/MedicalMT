# -*- coding: utf-8 -*-
"""
Display utilities for MedMT
"""
import sys
import time
from typing import Optional
import threading

class LoadingSpinner:
    def __init__(self, message: str = "Processing", char_set: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
        self.message = message
        self.char_set = char_set
        self.running = False
        self.spinner_thread = None
        
    def _spin(self):
        i = 0
        while self.running:
            spin_char = self.char_set[i % len(self.char_set)]
            sys.stdout.write(f"\r{spin_char} {self.message}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def start(self):
        self.running = True
        self.spinner_thread = threading.Thread(target=self._spin)
        self.spinner_thread.start()

    def stop(self, message: Optional[str] = None):
        self.running = False
        if self.spinner_thread:
            self.spinner_thread.join()
        if message:
            sys.stdout.write(f"\r✓ {message}\n")
        else:
            sys.stdout.write("\r")
        sys.stdout.flush()

class ProgressBar:
    def __init__(self, total: int, prefix: str = '', suffix: str = '', decimals: int = 1, length: int = 50, fill: str = '█', print_end: str = "\r"):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.print_end = print_end
        self.iteration = 0

    def print(self, iteration: Optional[int] = None):
        if iteration is not None:
            self.iteration = iteration
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.iteration / float(self.total)))
        filled_length = int(self.length * self.iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        print(f'\r{self.prefix} |{bar}| {percent}% {self.suffix}', end=self.print_end)
        if self.iteration == self.total:
            print()

    def increment(self):
        self.iteration += 1
        self.print()

def print_section_header(text: str, char: str = '='):
    """Print a section header with decorative characters"""
    width = min(80, max(50, len(text) + 4))
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")