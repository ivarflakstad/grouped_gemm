#include "grouped_gemm.hpp"

#include <torch/extension.h>
#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <memory>

#include <ck_tile/core.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/gemm.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/core/numeric/bfloat16.hpp>

template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float grouped_gemm(const std::vector<grouped_gemm_kargs>& gemm_descs,
                   const ck_tile::stream_config& s,
                   void* kargs_ptr) {
#if(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_MEMORY)
    // Memory friendly for Interwave scheduler
    constexpr ck_tile::index_t M_Tile = 128;
    constexpr ck_tile::index_t N_Tile = 32;
    constexpr ck_tile::index_t K_Tile = 64;

    constexpr ck_tile::index_t M_Warp = 4;
    constexpr ck_tile::index_t N_Warp = 1;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 8;

    constexpr bool DoubleSmemBuffer = false;
#endif
#if(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V3)
    // Compute friendly for Intrawave scheduler
    constexpr ck_tile::index_t M_Tile = 256;
    constexpr ck_tile::index_t N_Tile = 256;
    constexpr ck_tile::index_t K_Tile = 64;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 16;

    constexpr bool DoubleSmemBuffer = false;
#elif(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V4)
    // Compute friendly for Intrawave scheduler
    // Using the ping pong reader in the lds level
    constexpr ck_tile::index_t M_Tile = 256;
    constexpr ck_tile::index_t N_Tile = 256;
    constexpr ck_tile::index_t K_Tile = 32;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 16;

    constexpr bool DoubleSmemBuffer = true;
#endif

    constexpr bool kPadM = false;
    constexpr bool kPadN = false;
    constexpr bool kPadK = false;

    constexpr bool TransposeC = false;

    constexpr int kBlockPerCu                         = 1;
    constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    constexpr ck_tile::index_t TileParitionerM01      = 4;

    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
    using TilePartitioner = ck_tile::
        GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;
    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                 kPadN,
                                                                 kPadK,
                                                                 DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 CLayout,
                                                                 TransposeC>;
    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

    using BaseGemmPipeline = UNIVERSAL_GEMM_PIPELINE<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = gemm_descs[0].k_batch * K_Tile;
    const ck_tile::index_t K_split     = (gemm_descs[0].K + k_grain - 1) / k_grain * K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_,
                         const auto tail_number_,
                         const auto memory_operation_) {
        constexpr bool has_hot_loop_v   = has_hot_loop_.value;
        constexpr auto tail_number_v    = tail_number_.value;
        constexpr auto scheduler        = GEMM_PIPELINE_SCHEDULER;
        constexpr auto memory_operation = memory_operation_.value;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler,
                                                                           has_hot_loop_v,
                                                                           tail_number_v>;

        using GemmPipeline = GEMM_PIPELINE<UniversalGemmProblem>;
        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             CLayout,
                                             CDEElementWise,
                                             GemmPipelineProblem::kBlockSize,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             memory_operation>>;
        using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKargs(gemm_descs);
        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Kernel arguments not supported!");
        }

        constexpr dim3 blocks = Kernel::BlockSize();
        const dim3 grids      = Kernel::GridSize(gemm_descs);

        HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                            kargs.data(),
                                            get_workspace_size(gemm_descs),
                                            hipMemcpyHostToDevice,
                                            s.stream_id_));

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        ave_time =
            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<blocks.x, kBlockPerCu>(
                                       Kernel{},
                                       grids,
                                       blocks,
                                       0,
                                       ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                       gemm_descs.size()));

        return ave_time;
    };

    const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
        if(gemm_descs[0].k_batch == 1)
        {
            Run(has_hot_loop_,
                tail_number_,
                ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                           ck_tile::memory_operation_enum::set>{});
        }
        else
        {
            Run(has_hot_loop_,
                tail_number_,
                ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                           ck_tile::memory_operation_enum::atomic_add>{});
        }
    };

    BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);

    return ave_time;
}

template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          bool Persistent,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float invoke_gemm(
                  int group_count,
                  const std::vector<grouped_gemm_kargs>& args)
{
    // Workspace memory allocated to hold the gemm descriptions.
    ck_tile::DeviceMem gemm_workspace;
    gemm_workspace.Realloc(get_workspace_size(args));

    float ave_time = 0;
    if constexpr(!Persistent)
    {
        // Regular version of grouped gemm
        ave_time = grouped_gemm<ADataType,
                                BDataType,
                                DsDataType,
                                AccDataType,
                                CDataType,
                                ALayout,
                                BLayout,
                                DsLayout,
                                CLayout,
                                CDEElementWise>(
            args,
            ck_tile::stream_config{},
            gemm_workspace.GetDeviceBuffer());
    }
    else
    {
        // NOTE: With the persistent TileLoop kernel, we do not necessarily need to have
        // the gemm problems known on the host. Instead, we can just pass the pointer
        // to the kernel and let the workgroups figure out which tiles to work on.
        // This is useful when the gemm problems are generated dynamically.
        // In this example however, we generate the `kargs` using the known gemm_descs,
        // and copy the gemm descriptions to the device memory.
        // The contents of the memory pointed to by `kargs_ptr` pointer could be
        // written by e.g. another kernel from earlier stage.
        std::vector<ck_tile::GemmTransKernelArg> kargs;
        void* kargs_ptr   = gemm_workspace.GetDeviceBuffer();
        const bool splitk = args[0].k_batch > 1;
        for(const auto& arg : args)
        {
            kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<>{{arg.a_ptr},
                                                                  {arg.b_ptr},
                                                                  {/*arg.ds_ptr*/},
                                                                  arg.e_ptr,
                                                                  arg.M,
                                                                  arg.N,
                                                                  arg.K,
                                                                  {arg.stride_A},
                                                                  {arg.stride_B},
                                                                  {/*arg.stride_Ds*/},
                                                                  arg.stride_E,
                                                                  arg.k_batch});
        }
        const auto stream = ck_tile::stream_config{};
        HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                            kargs.data(),
                                            kargs.size() * sizeof(ck_tile::GemmTransKernelArg),
                                            hipMemcpyHostToDevice,
                                            stream.stream_id_));
        ave_time = grouped_gemm_tileloop<ALayout, BLayout, CLayout>(
            stream, group_count, kargs_ptr, splitk);
    }

    std::string op_name{"Grouped Gemm"};

    std::size_t flop = 0, num_btype = 0;
    for(int j = 0; j < group_count; ++j)
    {
        flop += std::size_t(2) * args[j].M * args[j].N * args[j].K;

        num_btype += sizeof(ADataType) * args[j].M * args[j].K +
                     sizeof(BDataType) * args[j].K * args[j].N +
                     sizeof(CDataType) * args[j].M * args[j].N;
    }

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
              << gb_per_sec << " GB/s, " << op_name << std::endl;

    return ave_time;
}

struct GemmProblemShape {
    ck::index_t m;
    ck::index_t n;
    ck::index_t k;
};

template<typename Layout>
struct IsRowMajor {
    static constexpr bool value = std::is_same_v<Layout, ck_tile::tensor_layout::gemm::RowMajor>;
};

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <
    typename ADataType,
    typename BDataType,
    typename AccDataType,
    typename CDataType,
    bool trans_a,
    bool trans_b
>
void ck_grouped_gemm(
    torch::Tensor a,
   	torch::Tensor b,
   	torch::Tensor c,
   	torch::Tensor batch_sizes,
   	GemmProblemShape gemm_problem_shape
) {
    using ALayout = GroupedGemmInputLayout<trans_a>;
    using BLayout = GroupedGemmInputLayout<trans_b>;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    //constexpr bool is_a_row_major = std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>::value;
    //constexpr bool is_b_row_major = std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>::value;

    int64_t num_experts = batch_sizes.size(0);
    std::vector<GemmProblemShape> problem_sizes_host(num_experts, gemm_problem_shape);
    std::vector<ck::index_t> lda_host(num_experts), ldb_host(num_experts), ldc_host(num_experts);
    int64_t elements_a = 0, elements_b = 0, elements_c = 0;
    std::vector<ADataType *> ptr_a_host(num_experts);
    std::vector<BDataType *> ptr_b_host(num_experts);
    std::vector<CDataType *> ptr_c_host(num_experts);

    auto stride_As = a.strides();
    auto stride_Bs = b.strides();
    auto stride_Cs = c.strides();

    std::vector<grouped_gemm_kargs> gemm_descs;
    gemm_descs.reserve(num_experts);

    constexpr ck::index_t kbatch = 1;

    for(int i = 0; i < num_experts; ++i) {
        auto& problem = problem_sizes_host[i];
        //problem.m = batch_sizes.data_ptr<int64_t>()[i];
        problem.k = (ck::index_t)batch_sizes.data_ptr<int64_t>()[i];

        ptr_a_host[i] = (ADataType*)a.data_ptr() + elements_a;
        ptr_b_host[i] = (BDataType*)b.data_ptr() + elements_b;
        ptr_c_host[i] = (CDataType*)c.data_ptr() + elements_c;

        lda_host[i] = ck_tile::get_default_stride(problem.m, problem.k, (int)stride_As[i], is_row_major(ALayout{}));
        ldb_host[i] = ck_tile::get_default_stride(problem.n, problem.k, (int)stride_Bs[i], is_row_major(BLayout{}));
        ldc_host[i] = ck_tile::get_default_stride(problem.m, problem.n, (int)stride_Cs[i], is_row_major(CLayout{}));

        elements_a += problem.m * problem.k;
        elements_b += problem.k * problem.n;
        elements_c += problem.m * problem.n;

        gemm_descs.push_back(grouped_gemm_kargs{
            ptr_a_host[i],
            ptr_b_host[i],
            ptr_c_host[i],
            kbatch,
            problem.m,
            problem.n,
            problem.k,
            lda_host[i],
            ldb_host[i],
            ldc_host[i],
        });
    }

    constexpr bool Persistent = false;

    invoke_gemm<ADataType,
                BDataType,
                ck_tile::tuple<>,
                AccDataType,
                CDataType,
                ALayout,
                BLayout,
                ck_tile::tuple<>,
                CLayout,
                Persistent>(num_experts, gemm_descs);

}

void call_grouped_gemm(
    torch::Tensor a,
   	torch::Tensor b,
   	torch::Tensor c,
   	torch::Tensor batch_sizes,
   	bool trans_a,
    bool trans_b
) {
    TORCH_CHECK(!(trans_a && trans_b));
    TORCH_CHECK(batch_sizes.ndimension() == 1);
    TORCH_CHECK(batch_sizes.scalar_type() == torch::kInt64);

    // We expected a CUDA tensor with two dimensions and shape
    // (tokens, hidden_in) for 'a'.
    TORCH_CHECK(a.is_cuda());
    TORCH_CHECK(b.is_cuda());
    TORCH_CHECK(c.is_cuda());
    TORCH_CHECK(a.ndimension() == 2);
    //TORCH_CHECK(a.scalar_type() == torch::kBFloat16);
    //TORCH_CHECK(b.scalar_type() == torch::kBFloat16);
    //TORCH_CHECK(c.scalar_type() == torch::kBFloat16);

    int hidden_in{}, hidden_out{}, k{};
    if (trans_a) {
      hidden_in = a.size(1);
      hidden_out = b.size(1);
      k = batch_sizes.size(1);

      TORCH_CHECK(b.ndimension() == 2);
      TORCH_CHECK(c.ndimension() == 3);
      TORCH_CHECK(b.size(0) == a.size(0));
      TORCH_CHECK(c.size(0) == batch_sizes.size(0));
      TORCH_CHECK(c.size(1) == hidden_in);
      TORCH_CHECK(c.size(2) == hidden_out);
    } else {
      TORCH_CHECK(b.ndimension() == 3);
      TORCH_CHECK(c.ndimension() == 2);

      // Validate the contraction dimensions match.
      int64_t tokens = a.size(0);
      int64_t num_experts = b.size(0);
      hidden_in = trans_b ? b.size(2) : b.size(1);
      hidden_out = trans_b ? b.size(1) : b.size(2);
      k = batch_sizes.size(0);
      TORCH_CHECK(hidden_in == a.size(1));

      // Validate that we have one size per expert.
      TORCH_CHECK(batch_sizes.size(0) == num_experts);
    }

    // NOTE: We support transposition through the 'trans_b' flag.
    TORCH_CHECK(a.is_contiguous());
    TORCH_CHECK(b.is_contiguous());
    TORCH_CHECK(c.is_contiguous());

    if (trans_a) {
        const auto gemm_shape = GemmProblemShape{hidden_in, hidden_out, k};
        ck_grouped_gemm<ADataType, BDataType, AccDataType, CDataType, true, false>(a, b, c, batch_sizes, gemm_shape);
        return;
    }
    if (trans_b) {
        const auto gemm_shape = GemmProblemShape{k, hidden_out, hidden_in};
        ck_grouped_gemm<ADataType, BDataType, AccDataType, CDataType, false, true>(a, b, c, batch_sizes, gemm_shape);
        return;
    }
    const auto gemm_shape = GemmProblemShape{k, hidden_out, hidden_in};
    ck_grouped_gemm<ADataType, BDataType, AccDataType, CDataType, false, false>(a, b, c, batch_sizes, gemm_shape);
    return;
}

void GroupedGemm(
    torch::Tensor a,
   	torch::Tensor b,
   	torch::Tensor c,
   	torch::Tensor batch_sizes,
   	bool trans_a,
    bool trans_b
) {
    TORCH_CHECK(!(trans_a && trans_b));
}
// Test function to verify the extension is working
torch::Tensor test_connection() {
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);
    return torch::ones({2, 2}, options) * 123.0f;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gmm", &GroupedGemm, "Grouped GEMM.");
    m.def("ck_gmm", &call_grouped_gemm, "CK Grouped GEMM.");
    m.def("test_connection", &test_connection, "Test that the extension is working");
}
