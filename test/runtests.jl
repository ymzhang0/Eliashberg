using Test
using Eliashberg

@testset "BCS Test" begin
    @test solve_bcs() == nothing
end

@testset "Eliashberg Test" begin
    @test solve_eliashberg() == nothing
end

