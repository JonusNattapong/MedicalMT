# -*- coding: utf-8 -*-
"""
ASCII banner component for MedMT
"""

import sys

def print_ascii_banner():
    """Print ASCII art banner for MedMT"""
    # Force UTF-8 output encoding
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    banner = """
    ▒███████▒ ▒█████   ███▄ ▄███▓ ▄▄▄▄     ██▓▄▄▄█████▓
    ▒ ▒ ▒ ▄▀░▒██▒  ██▒▓██▒▀█▀ ██▒▓█████▄ ▓██▒▓  ██▒ ▓▒
    ░ ▒ ▄▀▒░ ▒██░  ██▒▓██    ▓██░▒██▒ ▄██▒██▒▒ ▓██░ ▒░
      ▄▀▒   ░▒██   ██░▒██    ▒██ ▒██░█▀  ░██░░ ▓██▓ ░
    ▒███████▒░ ████▓▒░▒██▒   ░██▒░▓█  ▀█▓░██░  ▒██▒ ░
    ░▒▒ ▓░▒░▒░ ▒░▒░▒░ ░ ▒░   ░  ░░▒▓███▀▒░▓    ▒ ░░
    ░░▒ ▒ ░ ▒  ░ ▒ ▒░ ░  ░      ░▒░▒   ░  ▒ ░   ░
    ░ ░ ░ ░ ░░ ░ ░ ▒  ░      ░    ░    ░  ▒ ░ ░
      ░ ░        ░ ░         ░    ░       ░
    ░                               ░

    Medical Dialogue Machine Translation (Chinese→Thai)
                 Developer by JonusNattapong/zombit
               Powered by DeepSeek-Reasoner & Huggingface
         CC BY-SA-NC 4.0 | https://github.com/JonusNattapong/MedMT
    """
    print(banner)