using ControllerFormats, Test
import Aqua, ExplicitImports

@testset "ExplicitImports tests" begin
    ignores = (:rtoldefault, :Umlaut, :load, :load_file)
    @test isnothing(ExplicitImports.check_all_explicit_imports_are_public(ControllerFormats;
                                                                          ignore=ignores))
    ignores = (:Umlaut,)
    @test isnothing(ExplicitImports.check_all_explicit_imports_via_owners(ControllerFormats;
                                                                          ignore=ignores))
    ignores = (:ONNXCtx, :ProtoDecoder, :decode, :onnx_gemm)
    @test isnothing(ExplicitImports.check_all_qualified_accesses_are_public(ControllerFormats;
                                                                            ignore=ignores))
    ignores = (:ProtoDecoder, :decode)
    @test isnothing(ExplicitImports.check_all_qualified_accesses_via_owners(ControllerFormats;
                                                                            ignore=ignores))
    @test isnothing(ExplicitImports.check_no_implicit_imports(ControllerFormats))
    @test isnothing(ExplicitImports.check_no_self_qualified_accesses(ControllerFormats))
    @test isnothing(ExplicitImports.check_no_stale_explicit_imports(ControllerFormats))
end

@testset "Aqua tests" begin
    Aqua.test_all(ControllerFormats)
end
