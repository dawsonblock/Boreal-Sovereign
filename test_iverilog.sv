module test;
    logic [31:0] a;
    logic [10:0] b;
    always @* begin
        b = a[10:0];
    end
endmodule
