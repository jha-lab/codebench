module transposer_tb();

parameter IL = 4, FL = 16;
parameter row = 8, col = 8;

logic clk, reset;
logic input_ready;
logic output_taken;

logic signed [IL+FL-1:0][row-1:0][col-1:0] in;

logic [1:0] state;
logic signed [IL+FL-1:0][col-1:0][row-1:0] out;

transposer	#(.IL(IL), .FL(FL), .row(row), .col(col))	transposer_0
(
	.clk		(clk),
	.reset		(reset),
	.input_ready	(input_ready),
	.output_taken	(output_taken),
	.in		(in),
	.state		(state),
	.out		(out)
);

int i,j;
always_ff @(posedge clk) begin
        $display("reset=%b input_ready=%b output_taken=%b state=%b", reset, input_ready, output_taken, state);
        for (i = 0; i < row; i++) begin
		for (j = 0; j < col; j++) begin
                	$display("in[%d][%d]=%d", i, j, in[i][j]);
                	$display("out[%d][%d]=%d", j, i, out[j][i]);
        	end
	end
end

always #5 clk = !clk;

int p,q;
initial begin
        $dumpfile("transposer_tb.vcd");
        $dumpvars;
        clk = 0;
        reset = 1;
        #10
        reset = 0;
        input_ready = 1;
        for (p = 0; p < row; p++) begin
                for (q = 0; q < col; q++) begin
			in[p][q] = p + q;
		end
        end
	#10
	input_ready = 0;
	#50
	output_taken = 1;
	#10
	output_taken = 0;
	#10
	$finish;
end

endmodule
