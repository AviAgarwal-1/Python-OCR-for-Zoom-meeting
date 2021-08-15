#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

sys.path.insert(0,'/home/tekhawk/githubEAST/')

#import ocr
#%% EAST

#ocr.main()

#%% Reading
loc = r'/home/tekhawk/githubEAST/output/test-600.txt'
with open(loc) as file_in:
    lines = []
    for line in file_in:
        lines.append(line)
        