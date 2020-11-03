# accelerator
NN accelerator

batch normalization reference: https://wiseodd.github.io/techblog/2016/07/04/batchnorm/


1. All hardware description system verilog files for SPRING are in the `./RTL folder`.
2. All files needed for synthesis and reports generated are in the `./Synthesis folder`.

### How to run the Design Compiler simulation and generate area and power reports?

1. Go to the following website and source the source file. <https://researchcomputing.princeton.edu/faq/how-to-run-synopsys-and-c>
2. Go to the `./Synthesis` folder. Modify the `library file path` to your local directory.
3. For running the modules excepting dma, use `14nm_sg.tcl`. Modify the `top_module` variable to designate which module you want to run.
4. For running the dma module, use `dma.tcl`.
* Ps. To generate reports, check the file `report.tcl` in the directory `./Synthesis/script`. Need to create the report file path `./Synthesis/reports` first, and then establish those files `report_*.txt` in the folder.
