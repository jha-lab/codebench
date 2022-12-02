module im2col_tb();

parameter IL = 4, FL = 16;
parameter h = 512, w = 512;
parameter k = 16;
parameter h_width = $clog2(h);
parameter w_width = $clog2(w);
parameter k_width = $clog2(k);

logic clk, reset;
logic signed [IL+FL-1:0] im [h*w-1:0];
logic [h_width-1:0] im_h;
logic [w_width-1:0] im_w;
logic [k_width-1:0] k_h;
logic [k_width-1:0] k_w;
logic [k_width-1:0] stride;
logic input_ready;

logic signed [IL+FL-1:0] patch [k*k-1:0];
logic state;
logic done;

logic [$clog2(h*w)-1:0] index;

im2col im2col_0
(
	.clk		(clk),
	.reset		(reset),
	.im		(im),
	.im_h		(im_h),
	.im_w		(im_w),
	.k_h		(k_h),
	.k_w		(k_w),
	.stride		(stride),
	.input_ready	(input_ready),
	.patch		(patch),
	.state		(state),
	.done		(done)
);

always_ff @(posedge clk) begin
	$display("reset=%b", reset,
		 "im_h=%d im_w=%d", im_h, im_w,
		 "k_h=%d k_w=%d", k_h, k_w,
		 "stride=%d", stride,
		 "input_ready=%b", input_ready,
		 "patch=%d", patch,
		 "state=%b done=%b");
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
	#5
	reset = 1;
	#5
	reset = 0;
	#5
	for (index = 0; index < h*w; index = index + 1) begin
		im[index] = index;		
	end
	im_h = 256;
	im_w = 256;
	k_h = 3;
	k_w = 3;
	stride = 1;
	input_ready = 1;
	#1000
	$finish;
end

endmodule
