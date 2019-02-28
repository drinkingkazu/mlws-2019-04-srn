#!/usr/bin/python
import os
import sys
SPD_DIR = os.path.dirname(os.path.abspath(__file__))
SPD_DIR = os.path.dirname(SPD_DIR)
sys.path.insert(0, SPD_DIR)
from spd.flags import SPD_FLAGS

def main():
    flags = SPD_FLAGS()
    flags.parse_args()

if __name__ == '__main__':
    main()
