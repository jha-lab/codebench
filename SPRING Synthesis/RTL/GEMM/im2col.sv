module im2col
(
	clk,
	reset,
	im,
	im_h,
	im_w,
	k_h,
	k_w,
	stride,
	input_ready,
	patch,
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
input signed [IL+FL-1:0] im [h*w-1:0];
input [h_width-1:0] im_h;
input [w_width-1:0] im_w;
input [k_width-1:0] k_h;
input [k_width-1:0] k_w;
input [k_width-1:0] stride;
input input_ready;

output logic signed [IL+FL-1:0] patch [k*k-1:0];
output logic state;
output logic done;

logic [h_width-1:0] i;
logic [w_width-1:0] j;
logic [w_width-k_width-1:0] i_counter;
logic [k_width-1:0] index;

always_ff @(posedge clk) begin
	if (reset == 1) begin
		state <= 0;
	end
	else begin
		if (state == 0 && input_ready == 1) begin
			state <= 1;
		end
		if (state == 1 && done == 1) begin
			state <= 0;
		end
	end
end

always_ff @(posedge clk) begin
	if (reset == 1 || (state == 1 && done == 1)) begin
		j <= 0;
	end
	else begin
		if (state == 1) begin
			j <= j + stride;
		end
	end
end

always_ff @(posedge clk) begin
	if (reset == 1 || (state == 1 && done == 1)) begin
		i_counter <= 0;
	end
	else begin
		if (state == 1) begin
			if (i_counter == im_w / k_w - 1) begin
				i_counter <= 0;
			end
			else begin
				i_counter <= i_counter + 1;
			end
		end
	end
end

always_ff @(posedge clk) begin
	if (reset == 1 || (state == 1 && done == 1)) begin
		i <= 0;
	end
	else begin
		if (state == 1 && i_counter == im_w / k_w - 1) begin
			i <= i + stride;
		end
	end
end

always_comb begin
	if (state == 1) begin
		for (index = 0; index < k_h; index = index + 1) begin
			patch[(index+1)*k_w-1:index*k_w] = im[(i+index)*im_w+j+k_w-1:(i+index)*im_w+j];
		end 
	end	
end

always_comb begin
	if (state == 1 && i == (im_h/k_h)*k_h && j == (im_w/k_w)*k_w) begin
		done = 1;
	end
	else begin
		done = 0;
	end
end

endmodule
