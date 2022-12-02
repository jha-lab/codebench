module dataflow_tb();

parameter IL = 4, FL = 16;
parameter lane = 512;

logic clk, reset;
logic mode;
logic signed [IL+FL-1:0] in [lane-1:0][15:0];
logic signed [IL+FL-1:0] out [lane-1:0][15:0];

dataflow #(.IL(IL), .FL(FL), .lane(lane)) dataflow_0
(
	.clk		(clk),
	.reset		(reset),
	.mode		(mode),
	.in		(in),
	.out		(out)
);

int i,j,k;
always_ff @(posedge clk) begin
        $display("reset=%b mode=%b", reset, mode);
	for (i = 0; i < 16; i++) begin
        	$display("in[0][%d]=%d", i, in[0][i]);
	end
	for (j = 0; j < 5; j++) begin
		for (k = 0; k < 16; k++) begin
			$display("out[%d][%d]=%d", j, k, out[j][k]);		
		end
	end
end

always #5 clk = !clk;

int p,q;

initial begin
        $dumpfile("dataflow_tb.vcd");
        $dumpvars;

	clk = 0;
	reset = 1;
	#10
	reset = 0;
	mode = 0;
	for (p = 0; p < 512; p++) begin
		for (q = 0; q < 16; q++) begin
			in[p][q] = p+q;
		end
	end	
	#50
	mode = 1;
	#50

	$finish;
end
endmodule
