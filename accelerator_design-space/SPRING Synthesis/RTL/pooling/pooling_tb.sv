module pooling_tb();

parameter IL = 4, FL = 16;
parameter size = 4;
parameter width = $clog2(size);

logic clk;
logic reset;
logic [IL+FL-1:0] im [size-1:0];
logic input_ready;
logic output_taken;
logic [1:0] mode;
logic [IL+FL-1:0] om;
logic [1:0] state;

integer i;

pooling	pooling_0
(
	.clk		(clk),
        .reset		(reset),
        .im		(im),
        .input_ready	(input_ready),
        .output_taken	(output_taken),
        .mode		(mode),
        .om		(om),
        .state		(state)
);

always_ff @(posedge clk) begin
	$display("input_ready=%b output_taken=%b  mode=%b state=%b", input_ready, output_taken, mode, state);
	for (i = 0; i < size; i = i + 1) begin
		$display("im[%d]=%d", i, im[i]);
	end
	$display("om=%d", om);
end

initial begin
	forever begin
		clk = 0;
		#5
		clk = 1;
		#5
		clk = 0;
	end
end

initial begin
	reset = 0;
	#10
	reset = 1;
	#10
	reset = 0;;
	#10
	im[0] = 520;
	im[1] = 360;
	im[2] = 1378;
	im[3] = 280;
	input_ready = 1;
	mode = 2'b00;
	#10
	input_ready = 0;
	#100
	output_taken = 1;
	#10
	output_taken = 0;
	#10
	input_ready = 1;
	mode = 2'b01;
	#10
	input_ready = 0;
	#100
	output_taken = 1;
	#10
	output_taken = 0;
	#10
	input_ready = 1;
	mode = 2'b10;
	#10
	input_ready = 0;
	#100

	$finish;
end

endmodule
