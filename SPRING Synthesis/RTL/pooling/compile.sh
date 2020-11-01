#!/bin/sh

vcs -full64 -sverilog -timescale=1ns/10ps +define+CLOCK_PERIOD=0.5 -f filelist 
