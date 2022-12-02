# AccelBench Synthesis

Synthesis of the hardware modules used in AccelBench.

1. All hardware description System Verilog files for AccelBench are in the `./RTL` folder.
2. All files needed for synthesis and reports generated are in the `./Synthesis` folder.

### Run the Design Compiler simulation and generate area and power reports

1. Load license.
   * For tcsh or csh shell  
    `source /usr/licensed/synopsys/cshrc`  
    `source /usr/licensed/cadence/cshrc`  
   * For bash or sh shell  
    `. /usr/licensed/synopsys/profile`  
    `. /usr/licensed/cadence/profile`
2. Go to the `./Synthesis` folder. Modify the library file path `set LIBS` in `14nm_sg.tcl` and `dma.tcl` to your local `library` directory.
3. For running the modules except `dma`, use `14nm_sg.tcl`. Modify the top module in `set top_module` to designate which module you want to run.
4. For running the `dma` module, use `dma.tcl`.
5. Then, execute `run.cmd`.  
6. Check reports in `./Synthesis/reports`.
