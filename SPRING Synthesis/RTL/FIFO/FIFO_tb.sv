module fifo_tb();

parameter IL = 4, FL = 16;
parameter FIFO_DEPTH = 32;
parameter IN_BUS_WIDTH = IL + FL;
parameter FIFO_CNT_WIDTH = $clog2(FIFO_DEPTH);

logic clk;
logic reset;
logic [IN_BUS_WIDTH-1:0] data_in;
logic rd_en;
logic wr_en;
logic empty;
logic full;
logic [IN_BUS_WIDTH-1:0] data_out;
 
 
fifo #(.IL(IL), .FL(FL), .FIFO_DEPTH(FIFO_DEPTH), .IN_BUS_WIDTH(IN_BUS_WIDTH), .FIFO_CNT_WIDTH(FIFO_CNT_WIDTH)) fifo_0
(
    .clk		(clk),
    .reset		(reset),
    .data_in		(data_in),
    .rd_en		(rd_en),
    .wr_en		(wr_en),
    .empty		(empty),
    .full		(full),
    .data_out		(data_out)
);
 

always_ff @(posedge clk) begin
        $display("reset=%b rd_en=%b wr_en=%b empty=%b full=%b", reset, rd_en, wr_en, empty, full);
        $display("data_in=%d", data_in);
        $display("data_out=%d", data_out);
end

always #5 clk = !clk;

initial begin
        $dumpfile("FIFO_tb.vcd");
        $dumpvars;
        clk = 0;
        rd_en = 0;
	wr_en = 0;
        reset = 1;
        #10
        reset = 0;
        #10
	wr_en = 1;
	data_in = 101;
	#10
	wr_en = 0;
	rd_en = 1;
	#10
	rd_en = 0;
	wr_en = 1;
	data_in = 102;
	#10
	data_in = 103;
	#10
	data_in = 104;
	#10
	wr_en = 0;
	rd_en = 1;
	#50

        $finish;
end 

endmodule
