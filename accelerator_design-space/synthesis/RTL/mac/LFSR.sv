module LFSR
(
	clk,		//i
	reset,		//i
	data		//o
);

input clk, reset;
output logic [14:0] data;

logic feedback;

always_comb begin
	feedback = data[14] ^ data[0];
end

always_ff @(posedge clk) begin
	if (reset == 1) begin
		data <= 15'b111111111111111;
	end
	else begin
		data <= {data[13:1], feedback, data[14]};
	end
end

endmodule
