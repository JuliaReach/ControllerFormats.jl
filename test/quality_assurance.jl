using ControllerFormats, Test
import Aqua, ExplicitImports

@testset "ExplicitImports tests" begin
    ignores = (:rtoldefault,
               # false positive due to package extensions:
               :available_activations, :_id, :_relu, :_sigmoid)
    @test isnothing(ExplicitImports.check_all_explicit_imports_are_public(ControllerFormats;
                                                                          ignore=ignores))
    @test isnothing(ExplicitImports.check_all_explicit_imports_via_owners(ControllerFormats))
    ignores = (:ONNXCtx, :ProtoDecoder, :Umlaut, :decode, :load, :load_file,
               :onnx_gemm,
               # false positive due to package extensions:
               :_load_MAT, :_load_YAML, :_read_ONNX, :_id)
    @test isnothing(ExplicitImports.check_all_qualified_accesses_are_public(ControllerFormats;
                                                                            ignore=ignores))
    ignores = (:ProtoDecoder, :Umlaut, :decode)
    @test isnothing(ExplicitImports.check_all_qualified_accesses_via_owners(ControllerFormats;
                                                                            ignore=ignores))
    @test isnothing(ExplicitImports.check_no_implicit_imports(ControllerFormats))
    @test isnothing(ExplicitImports.check_no_self_qualified_accesses(ControllerFormats))
    @test isnothing(ExplicitImports.check_no_stale_explicit_imports(ControllerFormats))
end

@testset "Aqua tests" begin
    # Requires is only used in old versions
    @static if VERSION >= v"1.9"
        stale_deps = (ignore=[:Requires],)
    else
        stale_deps = true
    end

    Aqua.test_all(ControllerFormats; stale_deps=stale_deps)
end
