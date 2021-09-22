# accelerator
NN accelerator

batch normalization reference: https://wiseodd.github.io/techblog/2016/07/04/batchnorm/


1. All hardware description system verilog files for SPRING are in the `./RTL folder`.
2. All files needed for synthesis and reports generated are in the `./Synthesis folder`.

### How to run the Design Compiler simulation and generate area and power reports?
*Latest update by Vincent Li on 2021/9/22*

1. Go to the following website and source the source file. <https://researchcomputing.princeton.edu/faq/how-to-run-synopsys-and-c>
   * For tcsh or csh shell  
    `source /usr/licensed/synopsys/cshrc`  
    `source /usr/licensed/cadence/cshrc`  
   * For bash or sh shell  
    `. /usr/licensed/synopsys/profile`  
    `. /usr/licensed/cadence/profile`
2. Go to the `./Synthesis` folder. Modify the library file path `set LIBS` in `14nm_sg.tcl` and `dma.tcl` to your local `library` directory.
3. For running the modules except dma, use `14nm_sg.tcl`. Modify the top module in `set top_module` to designate which module you want to run.
4. For running the dma module, use `dma.tcl`.
5. Then, execute `run.cmd`.  
6. Check reports in `./Synthesis/reports`.

* Ps. To generate reports, check the file `report.tcl` in the directory `./Synthesis/script`. Need to create the report file path `./Synthesis/reports/{module_name}` first, and then establish those files `{module_name}.rpt` and `report_*.txt` in the folder.
