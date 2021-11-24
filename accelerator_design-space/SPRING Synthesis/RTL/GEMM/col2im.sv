module col2im
(
	clk,
	reset,
	patch,
	k_h,
	k_w,
	i,
	j,
	input_ready,
	output_taken,
	im,
	state,
	done
);

parameter IL = 4, FL = 16;
parameter h = 512, w = 512;
parameter k = 16;
parameter h_width = $clog2(h);
parameter w_width = $clog2(w);
parameter k_width = $clog2(k);

input clk, reset;
input signed [IL+FL-1:0] patch [k*k-1:0];
input [k_width-1:0] k_h;
input [k_width-1:0] k_w;
input [h_width-k_width-1:0] i;
input [w_width-k_width-1:0] j;

input input_ready;
input output_taken;

output logic signed [IL+FL-1:0] im [h*w-1:0];
output logic [1:0] state;
output logic done;

logic [h_width-k_width-1:0] ii;
logic [h_width-k_width-1:0] jj;
logic [k_width-1:0] index;

always_ff @(posedge clk) begin
	if (reset == 1) begin
		state <= 2'b00;
	end
	else begin
		if (state == 2'b00 && input_ready == 1) begin
			state <= 2'b01;
		end
		if (state == 2'b01 && done == 1) begin
			state <= 2'b10;
		end
		if (state == 2'b10 && output_taken == 1) begin
			state <= 2'b00;
		end
	end
end

always_ff @(posedge clk) begin
	if (reset == 1 || (state == 2'b10 && output_taken == 1)) begin
		jj <= 0;
		ii <= 0;
	end
	else begin
		if (state == 2'b01) begin
			if (jj == j) begin
				jj <= 0;
				ii <= i + 1;
			end
			else begin
				jj <= jj + 1;
			end
		end
	end
end

always_comb begin
	for (index = 0; index < k_h; index = index + 1) begin
		im[(ii+index)*(j+1)*k_w+(jj+1)*k_w-1:(ii+index)*(j+1)*k_w+jj*k_w] = patch[(index+1)*k_w-1:index*k_w];
	end
end

always_comb begin
	if (ii == i && jj == j) begin
		done = 1;
	end
	else begin
		done = 0;
	end
end

endmodule
