module loss_tb();

parameter IL = 4, FL = 16;
parameter size = 16;
parameter width = $clog2(size);

logic signed [IL+FL-1:0] yHat [size-1:0];
logic signed [IL+FL-1:0] y [size-1:0];
logic [width-1:0] num;
logic reset;
logic model;
logic [1:0] state;

logic signed [IL+FL-1:0] out;

logic clk;
logic input_ready;
logic output_taken;


integer i,j,k;

loss #(.IL(IL), .FL(FL), .size(size)) loss_0
(
	.clk		(clk),
        .reset		(reset),
        .model		(model),
        .yHat		(yHat),
        .y		(y),
        .num		(num),
        .input_ready	(input_ready),
        .output_taken	(output_taken),
        .state		(state),
        .out		(out)
);

always_ff @(posedge clk) begin
        $display("reset=%b model=%b num=%d out=%d input_ready=%b output_taken=%b state=%b", reset, model, num, out, input_ready, output_taken, state);
        for (i = 0; i < size; i = i + 1) begin
                $display("yHat[%d]=%d", i, yHat[i]);
                $display("y[%d]=%d", i, y[i]);
        end
end

always #5 clk = !clk;

initial begin
        $dumpfile("loss_tb.vcd");
        $dumpvars;
        clk = 0;
	reset = 1;
	#10
	reset = 0;
	input_ready = 1;
	num = 10;
	model = 1;
	for (j = 0; j < num; j = j + 1) begin
                yHat[j] = j;
                y[j] = 3;
        end
	#10
	input_ready = 0;
        #30
        output_taken = 1;
        #10
	output_taken = 0;
	#10
	input_ready = 1;
	model = 0;
	num = 5;
	for (k = 0; k < num; k = k + 1) begin
                yHat[k] = 2*k;
                y[k] = 5;
        end
	#10
	input_ready = 0;
	#30
	output_taken = 1;
	#10
	output_taken = 0;
	#10
	
        $finish;
end



endmodule
