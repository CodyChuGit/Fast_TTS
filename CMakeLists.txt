cmake_minimum_required(VERSION 3.20)

project(AudioCpp LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(AUDIOCPP_DEPLOYMENT_BUILD
    "Compile model specs into the runtime for self-contained deployments"
    OFF)

# Model specs remain editable runtime data in normal builds. The converter always
# carries the source catalog so the conversion binary is portable. Deployment builds
# additionally compile the catalog into engine_runtime as a fallback for CLI/server.
file(GLOB AUDIOCPP_MODEL_SPEC_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/model_specs/*.json")
set(AUDIOCPP_BUILTIN_MODEL_SPEC_ENTRIES "")
set(AUDIOCPP_CONVERTER_MODEL_SPEC_ENTRIES "")
foreach(AUDIOCPP_MODEL_SPEC_FILE IN LISTS AUDIOCPP_MODEL_SPEC_FILES)
    get_filename_component(AUDIOCPP_MODEL_SPEC_FAMILY "${AUDIOCPP_MODEL_SPEC_FILE}" NAME_WE)
    file(READ "${AUDIOCPP_MODEL_SPEC_FILE}" AUDIOCPP_MODEL_SPEC_JSON)
    string(APPEND AUDIOCPP_CONVERTER_MODEL_SPEC_ENTRIES
        "    {\"${AUDIOCPP_MODEL_SPEC_FAMILY}\", R\"AUDIOCPP_SPEC(${AUDIOCPP_MODEL_SPEC_JSON})AUDIOCPP_SPEC\"},\n")
    if (AUDIOCPP_DEPLOYMENT_BUILD)
        string(APPEND AUDIOCPP_BUILTIN_MODEL_SPEC_ENTRIES
            "    {\"${AUDIOCPP_MODEL_SPEC_FAMILY}\", R\"AUDIOCPP_SPEC(${AUDIOCPP_MODEL_SPEC_JSON})AUDIOCPP_SPEC\"},\n")
    endif()
endforeach()
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/model_specs.inc"
    CONTENT "${AUDIOCPP_BUILTIN_MODEL_SPEC_ENTRIES}")
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/gguf_converter_model_specs.inc"
    CONTENT "${AUDIOCPP_CONVERTER_MODEL_SPEC_ENTRIES}")

if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

if (NOT MSVC)
    set(AUDIOCPP_DEBUG_OPT_FLAGS "-O3 -g")
    set(CMAKE_C_FLAGS_DEBUG "${AUDIOCPP_DEBUG_OPT_FLAGS}" CACHE STRING "Optimized debug C flags" FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "${AUDIOCPP_DEBUG_OPT_FLAGS}" CACHE STRING "Optimized debug C++ flags" FORCE)
endif()

set(ENGINE_DEFAULT_ENABLE_CUDA OFF)
set(ENGINE_DEFAULT_ENABLE_HIP OFF)
set(ENGINE_DEFAULT_ENABLE_VULKAN OFF)
set(ENGINE_DEFAULT_ENABLE_METAL OFF)
set(ENGINE_DEFAULT_ENABLE_LLAMAFILE ON)
set(ENGINE_DEFAULT_ENABLE_CUDA_GRAPHS ON)
set(ENGINE_DEFAULT_ENABLE_NATIVE_CPU ON)
if (APPLE)
    set(ENGINE_DEFAULT_ENABLE_METAL ON)
endif()
if (DEFINED GGML_CUDA)
    set(ENGINE_DEFAULT_ENABLE_CUDA ${GGML_CUDA})
endif()
if (DEFINED GGML_HIP)
    set(ENGINE_DEFAULT_ENABLE_HIP ${GGML_HIP})
endif()
if (DEFINED GGML_VULKAN)
    set(ENGINE_DEFAULT_ENABLE_VULKAN ${GGML_VULKAN})
endif()
if (DEFINED GGML_METAL)
    set(ENGINE_DEFAULT_ENABLE_METAL ${GGML_METAL})
endif()
if (DEFINED GGML_LLAMAFILE)
    set(ENGINE_DEFAULT_ENABLE_LLAMAFILE ${GGML_LLAMAFILE})
endif()
if (DEFINED GGML_CUDA_GRAPHS)
    set(ENGINE_DEFAULT_ENABLE_CUDA_GRAPHS ${GGML_CUDA_GRAPHS})
endif()

option(ENGINE_ENABLE_CUDA "Build ggml with CUDA backend support" ${ENGINE_DEFAULT_ENABLE_CUDA})
option(ENGINE_ENABLE_HIP "Build ggml with HIP/ROCm backend support" ${ENGINE_DEFAULT_ENABLE_HIP})
option(ENGINE_ENABLE_VULKAN "Build ggml with Vulkan backend support" ${ENGINE_DEFAULT_ENABLE_VULKAN})
option(ENGINE_ENABLE_METAL "Build ggml with Metal backend support" ${ENGINE_DEFAULT_ENABLE_METAL})
option(ENGINE_ENABLE_LLAMAFILE "Build ggml with llamafile SGEMM support" ${ENGINE_DEFAULT_ENABLE_LLAMAFILE})
option(ENGINE_ENABLE_CUDA_GRAPHS "Enable ggml CUDA graphs support" ${ENGINE_DEFAULT_ENABLE_CUDA_GRAPHS})
option(ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS
    "Enable opt-in HIP optimizations validated only for Strix Halo (gfx1151)"
    OFF)
option(ENGINE_ENABLE_NATIVE_CPU "Build ggml CPU kernels with native host ISA flags" ${ENGINE_DEFAULT_ENABLE_NATIVE_CPU})
option(ENGINE_ENABLE_OPENMP "Build host code with OpenMP support" ON)
option(ENGINE_ENABLE_CPU_ALL_VARIANTS
    "Build CPU backends as dynamic libraries with per-ISA variants (for Docker/portable builds)"
    OFF)
option(ENGINE_BUILD_EXAMPLES "Build example binaries" OFF)
option(ENGINE_BUILD_TESTS "Build framework unit tests" OFF)
option(ENGINE_BUILD_WARMBENCH "Build warmbench helper binaries" OFF)
set(AUDIOCPP_GGML_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/external/ggml"
    CACHE PATH "ggml source tree to build with AudioCPP")

# HIP compiles ggml's CUDA backend sources as HIP code, so the two options drive the
# same ggml backend and cannot coexist: enabling both starts CUDA backend setup and
# HIP backend setup in the same configure, which fails in confusing ways downstream.
if (ENGINE_ENABLE_CUDA AND ENGINE_ENABLE_HIP)
    message(FATAL_ERROR
        "ENGINE_ENABLE_CUDA and ENGINE_ENABLE_HIP are mutually exclusive; "
        "configure with exactly one of them (e.g. -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_HIP=ON)")
endif()

set(AUDIOCPP_HIP_STRIX_HALO_OPTIMIZATIONS_ACTIVE OFF)
if (ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS)
    if (NOT ENGINE_ENABLE_HIP)
        message(FATAL_ERROR
            "ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=ON requires ENGINE_ENABLE_HIP=ON")
    endif()

    # Match ggml HIP target precedence: an explicit CMake HIP architecture wins,
    # followed by GPU_TARGETS and then the legacy AMDGPU_TARGETS spelling.
    if (CMAKE_HIP_ARCHITECTURES)
        set(AUDIOCPP_EFFECTIVE_HIP_TARGETS ${CMAKE_HIP_ARCHITECTURES})
        set(AUDIOCPP_EFFECTIVE_HIP_TARGET_SOURCE "CMAKE_HIP_ARCHITECTURES")
    elseif (GPU_TARGETS)
        set(AUDIOCPP_EFFECTIVE_HIP_TARGETS ${GPU_TARGETS})
        set(AUDIOCPP_EFFECTIVE_HIP_TARGET_SOURCE "GPU_TARGETS")
    elseif (AMDGPU_TARGETS)
        set(AUDIOCPP_EFFECTIVE_HIP_TARGETS ${AMDGPU_TARGETS})
        set(AUDIOCPP_EFFECTIVE_HIP_TARGET_SOURCE "AMDGPU_TARGETS")
    else()
        message(FATAL_ERROR
            "ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=ON requires exactly one explicit HIP target, gfx1151; "
            "set -DGPU_TARGETS=gfx1151 (no project-wide gfx1151 default is provided)")
    endif()

    list(LENGTH AUDIOCPP_EFFECTIVE_HIP_TARGETS AUDIOCPP_EFFECTIVE_HIP_TARGET_COUNT)
    if (NOT AUDIOCPP_EFFECTIVE_HIP_TARGET_COUNT EQUAL 1 OR
        NOT "${AUDIOCPP_EFFECTIVE_HIP_TARGETS}" STREQUAL "gfx1151")
        message(FATAL_ERROR
            "ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=ON requires the effective HIP target list to be exactly "
            "gfx1151; ${AUDIOCPP_EFFECTIVE_HIP_TARGET_SOURCE}='${AUDIOCPP_EFFECTIVE_HIP_TARGETS}'")
    endif()

    # hipBLASLt regressed decoder compute by roughly 3 seconds on gfx1151.
    # Keep the validated retained-graph path on rocBLAS; the generic HIP build
    # remains unchanged because this option is both target- and backend-gated.
    set(GGML_HIP_HIPBLASLT OFF CACHE BOOL
        "Use hipBLASLt instead of hipBLAS (rocBLAS) for HIP GEMM" FORCE)
    set(AUDIOCPP_HIP_STRIX_HALO_OPTIMIZATIONS_ACTIVE ON)
    message(STATUS "Strix Halo HIP optimizations enabled for gfx1151 with rocBLAS")
endif()

if (ENGINE_ENABLE_CUDA AND NOT ENGINE_ENABLE_HIP)
    enable_language(CUDA)
    # 12.0 is a floor, not a target: any 12.x or 13.x works (the Dockerfile pins
    # 12.9.0, and docs/build/windows.md ships a CUDA 13 package). The minimum is
    # enforced because an older toolkit configures happily and only fails at the
    # first .cu file, which reads as a code bug rather than a toolchain one: nvcc
    # < 11.6 cannot parse libstdc++ >= 11.3 headers ("parameter packs not expanded
    # with '...'" in <functional>), and distros still ship such pairings (Ubuntu
    # 22.04's nvidia-cuda-toolkit is 11.5). Fail here, with a pointer, instead.
    # The upper bound is the host compiler the chosen toolkit accepts, not CUDA
    # itself (e.g. CUDA 12.9 supports GCC <= 14), so it is left to the toolkit.
    find_package(CUDAToolkit 12.0 REQUIRED)
    if (NOT MSVC)
        set(CMAKE_CUDA_FLAGS_DEBUG "${AUDIOCPP_DEBUG_OPT_FLAGS}" CACHE STRING "Optimized debug CUDA flags" FORCE)
    endif()
endif()
if (ENGINE_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

if (ENGINE_ENABLE_CPU_ALL_VARIANTS)
    message(STATUS "CPU all variants mode: enabling shared libraries + backends as dynamic libs")
    set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
    set(GGML_BACKEND_DL ON CACHE BOOL "Build backends as dynamic libraries" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "Build all CPU backend variants" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "Build ggml CPU kernels with native host ISA flags" FORCE)
    # Ensure libraries in the same directory as the binary are found at runtime
    set(CMAKE_INSTALL_RPATH "\$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
else()
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "Build backends as dynamic libraries" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "Build all CPU backend variants" FORCE)
    set(GGML_NATIVE ${ENGINE_ENABLE_NATIVE_CPU} CACHE BOOL "Build ggml CPU kernels with native host ISA flags" FORCE)
endif()

set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Do not build upstream examples" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "Do not build upstream tests" FORCE)
set(GGML_ALL_WARNINGS OFF CACHE BOOL "Keep upstream warnings quiet in this sample" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "Do not probe for ccache in this local setup" FORCE)
set(GGML_CUDA ${ENGINE_ENABLE_CUDA} CACHE BOOL "Build ggml with CUDA backend support" FORCE)
set(GGML_HIP ${ENGINE_ENABLE_HIP} CACHE BOOL "Build ggml with HIP backend support" FORCE)
set(GGML_VULKAN ${ENGINE_ENABLE_VULKAN} CACHE BOOL "Build ggml with Vulkan backend support" FORCE)
set(GGML_METAL ${ENGINE_ENABLE_METAL} CACHE BOOL "Build ggml with Metal backend support" FORCE)
set(GGML_LLAMAFILE ${ENGINE_ENABLE_LLAMAFILE} CACHE BOOL "Build ggml with llamafile SGEMM support" FORCE)
set(GGML_CUDA_GRAPHS ${ENGINE_ENABLE_CUDA_GRAPHS} CACHE BOOL "Enable ggml CUDA graphs support" FORCE)
# HIP graphs are controlled by a separate upstream option that defaults to ON;
# keep it in sync so ENGINE_ENABLE_CUDA_GRAPHS=OFF also disables graphs on HIP builds
set(GGML_HIP_GRAPHS ${ENGINE_ENABLE_CUDA_GRAPHS} CACHE BOOL "Enable ggml HIP graphs support" FORCE)

set(SPM_ENABLE_SHARED OFF CACHE BOOL "Build sentencepiece shared libs" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "Do not build sentencepiece tests" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "Do not probe for tcmalloc" FORCE)

if (NOT EXISTS "${AUDIOCPP_GGML_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "AUDIOCPP_GGML_SOURCE_DIR does not point to a ggml source tree: ${AUDIOCPP_GGML_SOURCE_DIR}")
endif()
message(STATUS "Using ggml source: ${AUDIOCPP_GGML_SOURCE_DIR}")
add_subdirectory("${AUDIOCPP_GGML_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ggml")
add_subdirectory(external/sentencepiece EXCLUDE_FROM_ALL)

add_compile_options(
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wextra>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wpedantic>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-pedantic-errors>
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->
)

set(AUDIOCPP_MODEL_SET "full" CACHE STRING "Model composite to build: full, core, or custom")
set_property(CACHE AUDIOCPP_MODEL_SET PROPERTY STRINGS full core custom)
set(AUDIOCPP_MODELS "" CACHE STRING "Comma or semicolon separated model targets for AUDIOCPP_MODEL_SET=custom")
set(AUDIOCPP_LIBYAML_COMPILE_DEFINITIONS
    _POSIX_C_SOURCE=200809L
    YAML_DECLARE_STATIC=1
    YAML_VERSION_MAJOR=0
    YAML_VERSION_MINOR=2
    YAML_VERSION_PATCH=5
    YAML_VERSION_STRING=\"0.2.5\"
)

function(audiocpp_configure_runtime_object target_name)
    target_include_directories(${target_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/generated
        ${CMAKE_CURRENT_SOURCE_DIR}/external/sentencepiece/src
        ${CMAKE_CURRENT_SOURCE_DIR}/external/llama_tokenizer
        ${CMAKE_CURRENT_SOURCE_DIR}/external/cJSON
        ${CMAKE_CURRENT_SOURCE_DIR}/external/libyaml/include
    )
    target_compile_definitions(${target_name} PRIVATE ${AUDIOCPP_LIBYAML_COMPILE_DEFINITIONS})
    if (AUDIOCPP_HIP_STRIX_HALO_OPTIMIZATIONS_ACTIVE)
        target_compile_definitions(${target_name} PRIVATE ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=1)
    endif()
    target_link_libraries(${target_name} PRIVATE ggml sentencepiece)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
        if (MSVC)
            target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>)
        endif()
    endif()
endfunction()

function(audiocpp_add_model target_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDES;LOADERS;ALIASES;DEPENDS" ${ARGN})
    add_library(engine_model_${target_name} OBJECT ${ARG_SOURCES})
    set_target_properties(engine_model_${target_name} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    audiocpp_configure_runtime_object(engine_model_${target_name})
    set(AUDIOCPP_MODEL_TARGETS ${AUDIOCPP_MODEL_TARGETS} ${target_name} PARENT_SCOPE)
    set(AUDIOCPP_MODEL_${target_name}_INCLUDES "${ARG_INCLUDES}" PARENT_SCOPE)
    set(AUDIOCPP_MODEL_${target_name}_LOADERS "${ARG_LOADERS}" PARENT_SCOPE)
    set(AUDIOCPP_MODEL_${target_name}_DEPENDS "${ARG_DEPENDS}" PARENT_SCOPE)
    set(AUDIOCPP_MODEL_ALIAS_${target_name} ${target_name} PARENT_SCOPE)
    foreach(alias IN LISTS ARG_ALIASES)
        set(AUDIOCPP_MODEL_ALIAS_${alias} ${target_name} PARENT_SCOPE)
    endforeach()
endfunction()

add_library(engine_core OBJECT
    src/framework/assets/resource_bundle.cpp
    src/framework/model_spec/package.cpp
    src/framework/model_spec/options.cpp
    src/framework/model_spec/schema.cpp
    src/framework/assets/tensor_source.cpp
    src/framework/assets/weight_norm_folding.cpp
    src/framework/assets/torch_bin.cpp
    src/framework/core/module.cpp
    src/framework/core/backend.cpp
    src/framework/core/deferred_tensor_writer.cpp
    src/framework/core/execution_context.cpp
    src/framework/core/host_memory.cpp
    src/framework/runtime/model.cpp
    src/framework/model_spec/metadata.cpp
    src/framework/runtime/session.cpp
    src/framework/runtime/artifacts.cpp
    src/framework/runtime/cache.cpp
    src/framework/runtime/graph_executor.cpp
    src/framework/runtime/graph_optimizer.cpp
    src/framework/runtime/host_ops.cpp
    src/framework/runtime/kv_cache.cpp
    src/framework/runtime/bounded_static_kv_decode.cpp
    src/framework/runtime/options.cpp
    src/framework/runtime/session_base.cpp
    src/framework/runtime/workspace.cpp
    src/framework/midi/midi_file.cpp
    src/framework/io/filesystem.cpp
    src/framework/io/config.cpp
    src/framework/io/json.cpp
    src/framework/io/text.cpp
    src/framework/text/chinese_normalization.cpp
    src/framework/text/chunking.cpp
    src/framework/text/subtitle.cpp
    src/framework/text/text_normalization.cpp
    src/framework/text/unicode_normalization.cpp
    src/framework/io/yaml.cpp
    src/framework/io/binary.cpp
    src/framework/io/safetensors.cpp
    src/framework/debug/trace.cpp
    src/framework/debug/profiler.cpp
    src/framework/decoders/tdt_decoder_runner.cpp
    src/framework/decoders/tdt_decoder_greedy_customized.cpp
    src/framework/decoders/tdt_decoder_greedy_duration_loop.cpp
    src/framework/decoders/tdt_decoder_greedy_separate_heads.cpp
    src/framework/modules/linear_module.cpp
    src/framework/modules/packed_linear_weights.cpp
    src/framework/modules/primitive_modules.cpp
    src/framework/modules/activation_modules.cpp
    src/framework/modules/norm_modules.cpp
    src/framework/modules/lookup_modules.cpp
    src/framework/modules/structural_modules.cpp
    src/framework/modules/asr_helpers.cpp
    src/framework/modules/conv_modules.cpp
    src/framework/modules/zipformer_modules.cpp
    src/framework/modules/speech_encoders/whisper_embedding.cpp
    src/framework/modules/speech_encoders/whisper_frontend.cpp
    src/framework/modules/speech_encoders/campplus_encoder.cpp
    src/framework/modules/codecs/nemo_nano_codec.cpp
    src/framework/modules/pitch_extractors/rmvpe_pitch_extractor.cpp
    src/framework/modules/vocoders/bigvgan_vocoder.cpp
    src/framework/modules/vocoders/hift_vocoder.cpp
    src/framework/modules/speech_encoders/hubert_encoder.cpp
    src/framework/modules/speech_encoders/wav2vec2_bert_encoder.cpp
    src/framework/modules/speech_encoders/wavlm_encoder.cpp
    src/framework/modules/speech_encoders/sanm.cpp
    src/framework/modules/optimizations/fast_conv_modules.cpp
    src/framework/modules/optimizations/fast_projection_modules.cpp
    src/framework/modules/recurrent_modules.cpp
    src/framework/modules/streaming_conv_modules.cpp
    src/framework/modules/attention/positional_encoding.cpp
    src/framework/modules/attention/feed_forward.cpp
    src/framework/modules/attention/self_attention.cpp
    src/framework/modules/attention/longformer_attention.cpp
    src/framework/modules/attention/longformer_attention_gpu.cpp
    src/framework/modules/attention/common_relative_attention.cpp
    src/framework/modules/attention/relative_attention.cpp
    src/framework/modules/attention/scaled_dot_product_attention.cpp
    src/framework/modules/attention/grouped_query_attention.cpp
    src/framework/modules/attention/projected_grouped_self_attention.cpp
    src/framework/modules/attention/streaming_frame_attention_cache.cpp
    src/framework/modules/transformers/qwen_decoder.cpp
    src/framework/modules/transformers/qwen_causal_decoder.cpp
    src/framework/modules/transformers/qwen_causal_decode_runtime.cpp
    src/framework/modules/transformers/qwen3_vl_encoder_runtime.cpp
    src/framework/modules/attention/cross_attention.cpp
    src/framework/modules/attention/transformer_blocks.cpp
    src/framework/modules/conditioning_modules.cpp
    src/framework/modules/conformer_modules.cpp
    src/framework/modules/flow_sampler_runtime.cpp
    src/framework/modules/predictor_modules.cpp
    src/framework/modules/positional_modules.cpp
    src/framework/modules/text_encoders/t5_base_encoder.cpp
    src/framework/modules/text_encoders/t5_gemma_encoder.cpp
    src/framework/modules/transformers/gemma_decoder.cpp
    src/framework/modules/optimizations/fast_kv_modules.cpp
    src/framework/tokenizers/hf_tokenizer_json.cpp
    src/framework/tokenizers/llama_bpe.cpp
    external/llama_tokenizer/bpe-core.cpp
    external/llama_tokenizer/unicode.cpp
    external/llama_tokenizer/unicode-data.cpp
    src/framework/tokenizers/sentencepiece.cpp
    src/framework/audio/fft.cpp
    src/framework/audio/kaldi_fbank.cpp
    src/framework/audio/activity.cpp
    src/framework/audio/chunking.cpp
    src/framework/audio/conversion.cpp
    src/framework/audio/deepfilternet2.cpp
    src/framework/audio/dsp.cpp
    src/framework/audio/flashsr.cpp
    src/framework/audio/istft_graph.cpp
    src/framework/audio/mixing.cpp
    src/framework/audio/resampling.cpp
    src/framework/audio/output.cpp
    src/framework/audio/rnnoise.cpp
    src/framework/audio/utility_api.cpp
    src/framework/audio/waveform_ops.cpp
    src/framework/audio/wav_reader.cpp
    src/framework/audio/wav_writer.cpp
    src/framework/audio/zipenhancer.cpp
    src/framework/runtime/flow_kv_cache.cpp
    src/framework/modules/speaker_encoders/ecapa_tdnn_runtime.cpp
    src/framework/modules/speaker_encoders/titanet_runtime.cpp
    src/framework/codecs/fsq_audio_codec_runtime.cpp
    src/framework/codecs/mimi_codec_runtime.cpp
    src/framework/codecs/neural_audio.cpp
    src/framework/sampling/decode_modules.cpp
    src/framework/sampling/diffusion_math.cpp
    src/framework/sampling/hf_sampler.cpp
    src/framework/sampling/noise.cpp
    src/framework/sampling/torch_random.cpp
)
audiocpp_configure_runtime_object(engine_core)

# These two small built-in model paths are kept outside the selectable composite list for now.
add_library(engine_model_silero_vad OBJECT
    src/models/silero_vad/assets.cpp
    src/models/silero_vad/runtime.cpp
    src/models/silero_vad/session.cpp
)
set_target_properties(engine_model_silero_vad PROPERTIES EXCLUDE_FROM_ALL TRUE)
audiocpp_configure_runtime_object(engine_model_silero_vad)

if (MSVC)
    set_source_files_properties(
        src/community_models/inflect_v2/frontend.cpp
        PROPERTIES COMPILE_OPTIONS /utf-8
    )
endif()

audiocpp_add_model(qwen3_tts
    SOURCES
        src/models/qwen3_tts/assets.cpp
        src/models/qwen3_tts/tokenizer_text.cpp
        src/models/qwen3_tts/tokenizer_speech_encoder.cpp
        src/models/qwen3_tts/tokenizer_speech_decoder.cpp
        src/models/qwen3_tts/stream_decoder.cpp
        src/models/qwen3_tts/speaker_encoder.cpp
        src/models/qwen3_tts/talker.cpp
        src/models/qwen3_tts/prompt_tts_voice_clone.cpp
        src/models/qwen3_tts/prompt_tts_voice_design.cpp
        src/models/qwen3_tts/prompt_tts_custom_voice.cpp
        src/models/qwen3_tts/session.cpp
        src/models/qwen3_tts/loader.cpp
    INCLUDES
        engine/models/qwen3_tts/loader.h
    LOADERS
        engine::models::qwen3_tts::make_qwen3_tts_loader
)

audiocpp_add_model(qwen3_asr
    SOURCES
        src/models/qwen3_asr/assets.cpp
        src/models/qwen3_asr/tokenizer_text.cpp
        src/models/qwen3_asr/frontend_whisper.cpp
        src/models/qwen3_asr/audio_encoder.cpp
        src/models/qwen3_asr/thinker.cpp
        src/models/qwen3_asr/prompt_asr.cpp
        src/models/qwen3_asr/postprocess.cpp
        src/models/qwen3_asr/session.cpp
        src/models/qwen3_asr/loader.cpp
    INCLUDES
        engine/models/qwen3_asr/loader.h
    LOADERS
        engine::models::qwen3_asr::make_qwen3_asr_loader
    DEPENDS
        qwen3_forced_aligner
)

audiocpp_add_model(qwen3_forced_aligner
    SOURCES
        src/models/qwen3_forced_aligner/processor.cpp
        src/models/qwen3_forced_aligner/session.cpp
        src/models/qwen3_forced_aligner/loader.cpp
    INCLUDES
        engine/models/qwen3_forced_aligner/loader.h
    LOADERS
        engine::models::qwen3_forced_aligner::make_qwen3_forced_aligner_loader
    DEPENDS
        qwen3_asr
)

set(AUDIOCPP_ENABLED_MODELS "")
if (AUDIOCPP_MODEL_SET STREQUAL "full")
    set(AUDIOCPP_ENABLED_MODELS ${AUDIOCPP_MODEL_TARGETS})
elseif (AUDIOCPP_MODEL_SET STREQUAL "core")
    set(AUDIOCPP_ENABLED_MODELS "")
elseif (AUDIOCPP_MODEL_SET STREQUAL "custom")
    string(REPLACE "," ";" AUDIOCPP_REQUESTED_MODELS "${AUDIOCPP_MODELS}")
    foreach(requested_model IN LISTS AUDIOCPP_REQUESTED_MODELS)
        string(STRIP "${requested_model}" requested_model)
        if (requested_model STREQUAL "")
            continue()
        endif()
        if (DEFINED AUDIOCPP_MODEL_ALIAS_${requested_model})
            list(APPEND AUDIOCPP_ENABLED_MODELS ${AUDIOCPP_MODEL_ALIAS_${requested_model}})
        else()
            message(FATAL_ERROR "Unknown AUDIOCPP_MODELS entry: ${requested_model}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES AUDIOCPP_ENABLED_MODELS)
else()
    message(FATAL_ERROR "AUDIOCPP_MODEL_SET must be full, core, or custom")
endif()

set(AUDIOCPP_REGISTRY_INCLUDE_CONTENT "")
set(AUDIOCPP_REGISTRY_LOADER_CONTENT "")
foreach(model_name IN LISTS AUDIOCPP_ENABLED_MODELS)
    foreach(include_name IN LISTS AUDIOCPP_MODEL_${model_name}_INCLUDES)
        string(APPEND AUDIOCPP_REGISTRY_INCLUDE_CONTENT "#include \"${include_name}\"\n")
    endforeach()
    foreach(loader_name IN LISTS AUDIOCPP_MODEL_${model_name}_LOADERS)
        string(APPEND AUDIOCPP_REGISTRY_LOADER_CONTENT "        ${loader_name}(),\n")
    endforeach()
endforeach()

set(AUDIOCPP_LINKED_MODELS ${AUDIOCPP_ENABLED_MODELS})
set(AUDIOCPP_DEPENDENCY_SCAN_INDEX 0)
list(LENGTH AUDIOCPP_LINKED_MODELS AUDIOCPP_LINKED_MODEL_COUNT)
while(AUDIOCPP_DEPENDENCY_SCAN_INDEX LESS AUDIOCPP_LINKED_MODEL_COUNT)
    list(GET AUDIOCPP_LINKED_MODELS ${AUDIOCPP_DEPENDENCY_SCAN_INDEX} model_name)
    foreach(dependency IN LISTS AUDIOCPP_MODEL_${model_name}_DEPENDS)
        if (NOT dependency IN_LIST AUDIOCPP_MODEL_TARGETS)
            message(FATAL_ERROR "Unknown composite dependency for ${model_name}: ${dependency}")
        endif()
        if (NOT dependency IN_LIST AUDIOCPP_LINKED_MODELS)
            list(APPEND AUDIOCPP_LINKED_MODELS ${dependency})
        endif()
    endforeach()
    math(EXPR AUDIOCPP_DEPENDENCY_SCAN_INDEX "${AUDIOCPP_DEPENDENCY_SCAN_INDEX} + 1")
    list(LENGTH AUDIOCPP_LINKED_MODELS AUDIOCPP_LINKED_MODEL_COUNT)
endwhile()
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/model_registry_includes.inc"
    CONTENT "${AUDIOCPP_REGISTRY_INCLUDE_CONTENT}")
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/model_registry_loaders.inc"
    CONTENT "${AUDIOCPP_REGISTRY_LOADER_CONTENT}")

set(AUDIOCPP_RUNTIME_OBJECTS
    $<TARGET_OBJECTS:engine_core>
    $<TARGET_OBJECTS:engine_model_silero_vad>
)
foreach(model_name IN LISTS AUDIOCPP_LINKED_MODELS)
    list(APPEND AUDIOCPP_RUNTIME_OBJECTS $<TARGET_OBJECTS:engine_model_${model_name}>)
endforeach()

message(STATUS "audio.cpp model composite: ${AUDIOCPP_MODEL_SET} selected [${AUDIOCPP_ENABLED_MODELS}] linked [${AUDIOCPP_LINKED_MODELS}]")
add_library(engine_runtime STATIC
    src/framework/runtime/registry.cpp
    ${AUDIOCPP_RUNTIME_OBJECTS}
)

add_library(cjson_vendor STATIC
    external/cJSON/cJSON.c
)

target_include_directories(cjson_vendor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/cJSON
)

add_library(yaml_vendor STATIC
    external/libyaml/src/api.c
    external/libyaml/src/reader.c
    external/libyaml/src/scanner.c
    external/libyaml/src/parser.c
    external/libyaml/src/loader.c
)

target_include_directories(yaml_vendor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/libyaml/include
)

target_compile_definitions(yaml_vendor PUBLIC ${AUDIOCPP_LIBYAML_COMPILE_DEFINITIONS})

target_include_directories(engine_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(engine_runtime PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

target_include_directories(engine_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/sentencepiece/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/llama_tokenizer
)

target_link_libraries(engine_runtime PUBLIC ggml)
target_link_libraries(engine_runtime PRIVATE sentencepiece cjson_vendor yaml_vendor)
if (AUDIOCPP_HIP_STRIX_HALO_OPTIMIZATIONS_ACTIVE)
    target_compile_definitions(engine_runtime PRIVATE ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=1)
endif()
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(engine_runtime PRIVATE OpenMP::OpenMP_CXX)
    if (MSVC)
        # MSVC's default /openmp implements only OpenMP 2.0 and rejects the
        # '#pragma omp simd' directives in longformer_attention.cpp (error C7660).
        # /openmp:experimental enables the OpenMP 4.0 SIMD support; it overrides the
        # /openmp added by OpenMP::OpenMP_CXX above (harmless D9025 override notice).
        # Guard on CXX: nvcc does not forward a bare /openmp:experimental to the host
        # compiler and instead treats it as a second input file, which is fatal.
        target_compile_options(engine_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>)
    endif()
endif()

if (MSVC)
    target_compile_options(engine_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

if (ENGINE_ENABLE_CUDA AND NOT ENGINE_ENABLE_HIP)
    target_sources(engine_runtime PRIVATE
        src/framework/audio/istft_cuda_runtime.cu
        src/framework/sampling/torch_random_cuda_runtime.cu
    )
    set_source_files_properties(
        src/framework/audio/istft_cuda_runtime.cu
        src/framework/sampling/torch_random_cuda_runtime.cu
        PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    if (CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(engine_runtime PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set_target_properties(engine_runtime PROPERTIES CUDA_ARCHITECTURES native)
    endif()
    target_compile_definitions(engine_core PRIVATE ENGINE_HAS_CUDA_ISTFT=1 ENGINE_HAS_CUDA_TORCH_RANDOM=1)
    target_compile_definitions(engine_runtime PUBLIC ENGINE_HAS_CUDA_ISTFT=1 ENGINE_HAS_CUDA_TORCH_RANDOM=1)
    target_link_libraries(engine_runtime PRIVATE CUDA::cudart CUDA::cufft)
endif()

add_executable(audiocpp_cli
    app/cli/main.cpp
    app/cli/args.cpp
    app/cli/batch.cpp
    app/cli/request.cpp
    app/streaming/pcm_source.cpp
    app/streaming/streaming.cpp
    app/workflow/execution.cpp
    app/workflow/file_sink.cpp
    app/workflow/pipeline.cpp
    app/workflow/workflow.cpp
)

target_link_libraries(audiocpp_cli PRIVATE engine_runtime ggml)
# MinGW selects the wmain() entry point only when linked with -municode;
# without it the CLI fails to link (undefined reference / ld error 5).
# MSVC detects wmain natively, so this is MinGW-only. Link option only:
# -municode also defines UNICODE at compile time, so passing it link-only
# keeps the source compiled with no macro side effects.
if (MINGW)
    target_link_options(audiocpp_cli PRIVATE -municode)
endif()
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(audiocpp_cli PRIVATE OpenMP::OpenMP_CXX)
endif()

set(AUDIOCPP_UI_DIST "${CMAKE_CURRENT_SOURCE_DIR}/webui/native/dist/index.html")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${AUDIOCPP_UI_DIST}")
if (EXISTS "${AUDIOCPP_UI_DIST}")
    file(READ "${AUDIOCPP_UI_DIST}" AUDIOCPP_UI_HEX HEX)
else()
    string(HEX "<!doctype html><title>audio.cpp</title><p>Embedded WebUI assets are unavailable.</p>" AUDIOCPP_UI_HEX)
endif()
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," AUDIOCPP_UI_BYTES "${AUDIOCPP_UI_HEX}")
string(REPLACE "," ",\n" AUDIOCPP_UI_BYTES "${AUDIOCPP_UI_BYTES}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/app/server/audiocpp_ui_asset.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/audiocpp_ui_asset.h"
    @ONLY
)

set(AUDIOCPP_DEMO_VOICE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/webui/native/demo_voices")
macro(audiocpp_embed_demo_voice_file INPUT_FILE OUTPUT_VARIABLE)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${INPUT_FILE}")
    file(READ "${INPUT_FILE}" _audiocpp_demo_voice_hex HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," ${OUTPUT_VARIABLE} "${_audiocpp_demo_voice_hex}")
    string(REPLACE "," ",\n" ${OUTPUT_VARIABLE} "${${OUTPUT_VARIABLE}}")
endmacro()
audiocpp_embed_demo_voice_file(
    "${AUDIOCPP_DEMO_VOICE_DIR}/demo_1_man.wav"
    AUDIOCPP_DEMO_1_MAN_BYTES)
audiocpp_embed_demo_voice_file(
    "${AUDIOCPP_DEMO_VOICE_DIR}/demo_2_man.wav"
    AUDIOCPP_DEMO_2_MAN_BYTES)
audiocpp_embed_demo_voice_file(
    "${AUDIOCPP_DEMO_VOICE_DIR}/demo_3_woman.wav"
    AUDIOCPP_DEMO_3_WOMAN_BYTES)
audiocpp_embed_demo_voice_file(
    "${AUDIOCPP_DEMO_VOICE_DIR}/demo_4_woman.wav"
    AUDIOCPP_DEMO_4_WOMAN_BYTES)
audiocpp_embed_demo_voice_file(
    "${AUDIOCPP_DEMO_VOICE_DIR}/prompt_text"
    AUDIOCPP_DEMO_PROMPT_TEXT_BYTES)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/app/server/audiocpp_demo_voice_assets.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/audiocpp_demo_voice_assets.h"
    @ONLY
)

add_executable(audiocpp_server
    app/server/main.cpp
    app/server/base64.cpp
    app/server/character.cpp
    app/server/chat_hygiene.cpp
    app/server/config.cpp
    app/server/filler.cpp
    app/server/voice.cpp
    app/server/voice_stability.cpp
    app/server/http.cpp
    app/server/llm_client.cpp
    app/server/llm_manager.cpp
    app/server/mcp.cpp
    app/server/segmenter.cpp
    app/server/multipart.cpp
    app/server/runtime.cpp
    app/server/ui_assets.cpp
    app/cli/args.cpp
    app/cli/request.cpp
    app/streaming/pcm_source.cpp
    app/streaming/streaming.cpp
)

target_link_libraries(audiocpp_server PRIVATE engine_runtime ggml)
target_include_directories(audiocpp_server PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
if (ENGINE_ENABLE_CUDA AND NOT ENGINE_ENABLE_HIP)
    target_sources(audiocpp_server PRIVATE app/server/cuda_keepalive.cu)
    set_source_files_properties(
        app/server/cuda_keepalive.cu
        PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    if (CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(audiocpp_server PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set_target_properties(audiocpp_server PROPERTIES CUDA_ARCHITECTURES native)
    endif()
    target_compile_definitions(audiocpp_server PRIVATE AUDIOCPP_SERVER_HAS_CUDA_KEEPALIVE=1)
    target_include_directories(audiocpp_server PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
endif()
if (WIN32)
    target_link_libraries(audiocpp_server PRIVATE ws2_32)
endif()
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(audiocpp_server PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(audiocpp_gguf
    app/gguf/main.cpp
)
target_link_libraries(audiocpp_gguf PRIVATE engine_runtime ggml)
target_include_directories(audiocpp_gguf PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(audiocpp_gguf PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(model_perf
    tests/perf/model_perf.cpp
    app/cli/args.cpp
    app/cli/request.cpp
)
target_link_libraries(model_perf PRIVATE engine_runtime ggml)
target_include_directories(model_perf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(model_perf PRIVATE OpenMP::OpenMP_CXX)
endif()




if (ENGINE_ENABLE_OPENMP)
endif()

# Only build this parity check when the vibevoice model is linked, since it
# directly references model-internal symbols (load_vibevoice_assets etc.).
if (vibevoice IN_LIST AUDIOCPP_LINKED_MODELS)
    if (ENGINE_ENABLE_OPENMP)
    endif()
endif()

if (ENGINE_BUILD_TESTS)

    if (ENGINE_ENABLE_OPENMP)
    endif()


    if (ENGINE_ENABLE_OPENMP)
    endif()


    if (ENGINE_ENABLE_OPENMP)
    endif()


    if (ENGINE_ENABLE_OPENMP)
    endif()
endif()

if (ENGINE_BUILD_WARMBENCH)
    function(add_engine_warmbench target_name source_file)
        add_executable(${target_name}
            ${source_file}
        )
        target_link_libraries(${target_name} PRIVATE engine_runtime ggml)
        if (ENGINE_ENABLE_OPENMP)
            target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endfunction()

    add_engine_warmbench(qwen3_asr_warm_bench tests/qwen3_asr/qwen3_asr_warm_bench.cpp)
    add_engine_warmbench(qwen3_forced_aligner_warm_bench tests/qwen3_forced_aligner/qwen3_forced_aligner_warm_bench.cpp)
    add_engine_warmbench(qwen3_tts_warm_bench tests/qwen3_tts/qwen3_tts_warm_bench.cpp)
    add_engine_warmbench(qwen3_tts_stream_latency benchmarks/stream_latency/main.cpp)
    add_engine_warmbench(qwen3_tts_streaming_integration tests/qwen3_tts/qwen3_tts_streaming_integration.cpp)
    add_engine_warmbench(silero_vad_warm_bench tests/silero_vad/silero_vad_warm_bench.cpp)
endif()

if (ENGINE_BUILD_TESTS)
    enable_testing()

    set(ENGINE_UNITTEST_ASSET_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests/assets")

    function(add_engine_unittest target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE engine_runtime ggml)
    endfunction()

    add_executable(sentencepiece_tokenizer1_test
        tests/unittests/test_sentencepiece_tokenizer1.cpp
    )

    target_link_libraries(sentencepiece_tokenizer1_test PRIVATE engine_runtime ggml)
    target_compile_definitions(sentencepiece_tokenizer1_test PRIVATE
        ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
    )

    add_test(
        NAME sentencepiece_tokenizer1_test
        COMMAND sentencepiece_tokenizer1_test
    )

    add_engine_unittest(citrinet_tokenizer_decode_test tests/unittests/test_citrinet_tokenizer_decode.cpp)
    target_compile_definitions(citrinet_tokenizer_decode_test PRIVATE
        ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
    )

    add_test(
        NAME citrinet_tokenizer_decode_test
        COMMAND citrinet_tokenizer_decode_test
    )

    add_engine_unittest(audio_dsp_test tests/unittests/test_audio_dsp.cpp)

    add_test(
        NAME audio_dsp_test
        COMMAND audio_dsp_test
    )


    add_engine_unittest(midi_file_test tests/unittests/test_midi_file.cpp)

    add_test(
        NAME midi_file_test
        COMMAND midi_file_test
    )

    add_engine_unittest(wav_reader_test tests/unittests/test_wav_reader.cpp)

    add_test(
        NAME wav_reader_test
        COMMAND wav_reader_test
    )

    add_engine_unittest(wav_reader_chunk_bounds_test tests/unittests/test_wav_reader_chunk_bounds.cpp)

    add_test(
        NAME wav_reader_chunk_bounds_test
        COMMAND wav_reader_chunk_bounds_test
    )

    add_engine_unittest(hf_tokenizer_vocab_bounds_test tests/unittests/test_hf_tokenizer_vocab_bounds.cpp)

    add_test(
        NAME hf_tokenizer_vocab_bounds_test
        COMMAND hf_tokenizer_vocab_bounds_test
    )

    add_engine_unittest(safetensors_offsets_test tests/unittests/test_safetensors_offsets.cpp)

    add_test(
        NAME safetensors_offsets_test
        COMMAND safetensors_offsets_test
    )

    add_engine_unittest(audio_chunking_test tests/unittests/test_audio_chunking.cpp)

    add_test(
        NAME audio_chunking_test
        COMMAND audio_chunking_test
    )





    add_engine_unittest(chinese_normalization_test tests/unittests/test_chinese_normalization.cpp)

    add_test(
        NAME chinese_normalization_test
        COMMAND chinese_normalization_test
    )

    add_engine_unittest(index_tts_tn_corpus_test tests/unittests/test_index_tts_tn_corpus.cpp)
    target_compile_definitions(index_tts_tn_corpus_test PRIVATE
        ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
    )

    add_test(
        NAME index_tts_tn_corpus_test
        COMMAND index_tts_tn_corpus_test
    )



    add_engine_unittest(text_normalization_test tests/unittests/test_text_normalization.cpp)

    add_test(
        NAME text_normalization_test
        COMMAND text_normalization_test
    )




    add_engine_unittest(subtitle_formatter_test tests/unittests/test_subtitle_formatter.cpp)

    add_test(
        NAME subtitle_formatter_test
        COMMAND subtitle_formatter_test
    )

    add_engine_unittest(rnnoise_utility_test tests/unittests/test_rnnoise_utility.cpp)
    target_compile_definitions(rnnoise_utility_test PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
        ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
    )

    add_test(
        NAME rnnoise_utility_test
        COMMAND rnnoise_utility_test
    )

    add_engine_unittest(flashsr_utility_test tests/unittests/test_flashsr_utility.cpp)
    target_compile_definitions(flashsr_utility_test PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
        ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
    )

    add_test(
        NAME flashsr_utility_test
        COMMAND flashsr_utility_test
    )

    add_engine_unittest(zipenhancer_utility_test tests/unittests/test_zipenhancer_utility.cpp)
    target_compile_definitions(zipenhancer_utility_test PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
        ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
    )

    add_test(
        NAME zipenhancer_utility_test
        COMMAND zipenhancer_utility_test
    )

    add_engine_unittest(audio_utility_api_test tests/unittests/test_audio_utility_api.cpp)
    target_compile_definitions(audio_utility_api_test PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(
        NAME audio_utility_api_test
        COMMAND audio_utility_api_test
    )

    add_engine_unittest(torch_random_test tests/unittests/test_torch_random.cpp)

    add_test(
        NAME torch_random_test
        COMMAND torch_random_test
    )

    add_engine_unittest(backend_device_resolution_test tests/unittests/test_backend_device_resolution.cpp)
    target_include_directories(backend_device_resolution_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME backend_device_resolution_test
        COMMAND backend_device_resolution_test
    )

    add_executable(streaming_audio_input_test
        tests/unittests/test_streaming_audio_input.cpp
        app/streaming/pcm_source.cpp
        app/streaming/streaming.cpp
    )
    target_link_libraries(streaming_audio_input_test PRIVATE engine_runtime ggml)
    target_include_directories(streaming_audio_input_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME streaming_audio_input_test
        COMMAND streaming_audio_input_test
    )

    add_executable(http_live_body_test
        tests/unittests/test_http_live_body.cpp
        app/server/http.cpp
    )
    target_link_libraries(http_live_body_test PRIVATE engine_runtime ggml)
    if(WIN32)
        target_link_libraries(http_live_body_test PRIVATE ws2_32)
    endif()
    target_include_directories(http_live_body_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME http_live_body_test
        COMMAND http_live_body_test
    )
    # This test talks to a real loopback socket, so a framing regression could
    # block forever rather than fail. Bound it so CI reports a failure instead.
    set_tests_properties(http_live_body_test PROPERTIES TIMEOUT 120)

    add_engine_unittest(partial_text_render_test tests/unittests/test_partial_text_render.cpp)

    add_test(
        NAME partial_text_render_test
        COMMAND partial_text_render_test
    )

    add_engine_unittest(hf_sampler_test tests/unittests/test_hf_sampler.cpp)

    add_test(
        NAME hf_sampler_test
        COMMAND hf_sampler_test
    )

    add_engine_unittest(diffusion_math_test tests/unittests/test_diffusion_math.cpp)

    add_test(
        NAME diffusion_math_test
        COMMAND diffusion_math_test
    )

    add_engine_unittest(encoder_module_test tests/unittests/test_encoder_modules.cpp)

    add_test(
        NAME encoder_module_test
        COMMAND encoder_module_test
    )

    add_engine_unittest(
        qwen_decoder_packed_projection_test
        tests/unittests/test_qwen_decoder_packed_projections.cpp
    )


    add_engine_unittest(conv_transpose_fast_path_test tests/unittests/test_conv_transpose_fast_path.cpp)

    add_test(
        NAME conv_transpose_fast_path_test
        COMMAND conv_transpose_fast_path_test
    )



    add_engine_unittest(depthwise_conv1d_lowering_test tests/unittests/test_depthwise_conv1d_lowering.cpp)

    add_test(
        NAME depthwise_conv1d_lowering_test
        COMMAND depthwise_conv1d_lowering_test
    )

    add_engine_unittest(conv_lowering_matrix_test tests/unittests/test_conv_lowering_matrix.cpp)
    if (ENGINE_ENABLE_HIP)
        target_compile_definitions(conv_lowering_matrix_test PRIVATE ENGINE_TEST_ENABLE_HIP=1)
    endif()

    add_test(
        NAME conv_lowering_matrix_test
        COMMAND conv_lowering_matrix_test
    )





    add_engine_unittest(qwen3_forced_aligner_processor_test tests/unittests/test_qwen3_forced_aligner_processor.cpp)
    target_include_directories(qwen3_forced_aligner_processor_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME qwen3_forced_aligner_processor_test
        COMMAND qwen3_forced_aligner_processor_test
    )

    add_engine_unittest(qwen3_tts_stream_decoder_test tests/unittests/test_qwen3_tts_stream_decoder.cpp)
    target_include_directories(qwen3_tts_stream_decoder_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME qwen3_tts_stream_decoder_test
        COMMAND qwen3_tts_stream_decoder_test
    )







    set(
        AUDIOCPP_FUN_ASR_NANO_TEST_MODEL
        ""
        CACHE PATH
        "Optional local Fun-ASR-Nano model directory for full runtime parity tests"
    )
    set(
        AUDIOCPP_FUN_ASR_NANO_TEST_GGUF
        ""
        CACHE FILEPATH
        "Optional standalone Fun-ASR-Nano GGUF for full CLI transcript tests"
    )
    if (AUDIOCPP_FUN_ASR_NANO_TEST_MODEL)
        if (GGML_CUDA)
        endif()

        if (GGML_CUDA)
        endif()
    endif()
    if (AUDIOCPP_FUN_ASR_NANO_TEST_GGUF)
        add_test(
            NAME fun_asr_nano_gguf_cli_cpu
            COMMAND audiocpp_cli
                --task asr
                --family fun_asr_nano
                --model ${AUDIOCPP_FUN_ASR_NANO_TEST_GGUF}
                --backend cpu
                --audio ${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav
        )
        set_tests_properties(
            fun_asr_nano_gguf_cli_cpu
            PROPERTIES PASS_REGULAR_EXPRESSION
                "text_output=Some call me nature, others call me mother nature"
        )
        if (GGML_CUDA)
            add_test(
                NAME fun_asr_nano_gguf_cli_cuda
                COMMAND audiocpp_cli
                    --task asr
                    --family fun_asr_nano
                    --model ${AUDIOCPP_FUN_ASR_NANO_TEST_GGUF}
                    --backend cuda
                    --audio ${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav
            )
            set_tests_properties(
                fun_asr_nano_gguf_cli_cuda
                PROPERTIES PASS_REGULAR_EXPRESSION
                    "text_output=Some call me nature, others call me mother nature"
            )
            add_test(
                NAME fun_asr_nano_gguf_cli_cuda_shared_q8
                COMMAND audiocpp_cli
                    --task asr
                    --family fun_asr_nano
                    --model ${AUDIOCPP_FUN_ASR_NANO_TEST_GGUF}
                    --backend cuda
                    --session-option fun_asr_nano.weight_type=q8_0
                    --audio ${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav
            )
            set_tests_properties(
                fun_asr_nano_gguf_cli_cuda_shared_q8
                PROPERTIES PASS_REGULAR_EXPRESSION
                    "text_output=Some call me nature, others call me mother nature"
            )
            add_test(
                NAME fun_asr_nano_gguf_cli_best_shared_q8
                COMMAND audiocpp_cli
                    --task asr
                    --family fun_asr_nano
                    --model ${AUDIOCPP_FUN_ASR_NANO_TEST_GGUF}
                    --backend best
                    --session-option fun_asr_nano.weight_type=q8_0
                    --audio ${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav
            )
            set_tests_properties(
                fun_asr_nano_gguf_cli_best_shared_q8
                PROPERTIES PASS_REGULAR_EXPRESSION
                    "text_output=Some call me nature, others call me mother nature"
            )
        endif()
    endif()

    if (AUDIOCPP_FUN_ASR_NANO_TEST_MODEL)
        if (GGML_CUDA)
        endif()
    endif()

    if (AUDIOCPP_FUN_ASR_NANO_TEST_MODEL)
        if (GGML_CUDA)
        endif()
    endif()

    add_engine_unittest(tdt_decoder_duration_loop_test tests/unittests/test_tdt_decoder_duration_loop.cpp)
    target_include_directories(tdt_decoder_duration_loop_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME tdt_decoder_duration_loop_test
        COMMAND tdt_decoder_duration_loop_test
    )

    add_engine_unittest(scaled_dot_product_attention_test tests/unittests/test_scaled_dot_product_attention.cpp)

    add_test(
        NAME scaled_dot_product_attention_test
        COMMAND scaled_dot_product_attention_test
    )



    add_executable(cli_request_options_test
        tests/unittests/test_cli_request_options.cpp
        app/cli/args.cpp
        app/cli/request.cpp
    )
    target_link_libraries(cli_request_options_test PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(cli_request_options_test PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_test(
        NAME cli_request_options_test
        COMMAND cli_request_options_test
    )

    add_executable(server_multipart_test
        tests/unittests/test_server_multipart.cpp
        app/server/multipart.cpp
    )
    target_include_directories(server_multipart_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)

    add_test(
        NAME server_multipart_test
        COMMAND server_multipart_test
    )

    add_executable(server_segmenter_test
        tests/unittests/test_server_segmenter.cpp
        app/server/segmenter.cpp
    )
    target_include_directories(server_segmenter_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)

    add_test(
        NAME server_segmenter_test
        COMMAND server_segmenter_test
    )

    add_executable(server_llm_client_test
        tests/unittests/test_server_llm_client.cpp
        app/server/llm_client.cpp
    )
    target_include_directories(server_llm_client_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)
    target_link_libraries(server_llm_client_test PRIVATE engine_runtime ggml)
    if(WIN32)
        target_link_libraries(server_llm_client_test PRIVATE ws2_32)
    endif()

    add_test(
        NAME server_llm_client_test
        COMMAND server_llm_client_test
    )

    add_executable(server_mcp_test
        tests/unittests/test_server_mcp.cpp
        app/server/mcp.cpp
    )
    target_include_directories(server_mcp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)
    target_link_libraries(server_mcp_test PRIVATE engine_runtime ggml)

    add_test(
        NAME server_mcp_test
        COMMAND server_mcp_test
    )

    add_executable(server_character_test
        tests/unittests/test_server_character.cpp
        app/server/character.cpp
    )
    target_include_directories(server_character_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)
    target_link_libraries(server_character_test PRIVATE engine_runtime ggml)

    add_test(
        NAME server_character_test
        COMMAND server_character_test
    )

    add_executable(server_chat_hygiene_test
        tests/unittests/test_server_chat_hygiene.cpp
        app/server/chat_hygiene.cpp
    )
    target_include_directories(server_chat_hygiene_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)

    add_test(
        NAME server_chat_hygiene_test
        COMMAND server_chat_hygiene_test
    )

    add_executable(server_voice_stability_test
        tests/unittests/test_server_voice_stability.cpp
        app/server/voice_stability.cpp
    )
    target_include_directories(server_voice_stability_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)

    add_test(
        NAME server_voice_stability_test
        COMMAND server_voice_stability_test
    )

    add_executable(server_base64_test
        tests/unittests/test_server_base64.cpp
        app/server/base64.cpp
    )
    target_include_directories(server_base64_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)

    add_test(
        NAME server_base64_test
        COMMAND server_base64_test
    )

    add_executable(server_busy_guard_test
        tests/unittests/test_server_busy_guard.cpp
    )
    target_include_directories(server_busy_guard_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)
    find_package(Threads REQUIRED)
    target_link_libraries(server_busy_guard_test PRIVATE Threads::Threads)

    add_test(
        NAME server_busy_guard_test
        COMMAND server_busy_guard_test
    )
    # The suite deliberately drives threads against a held lock; bound it so a
    # regression that reintroduces unbounded queuing fails instead of hanging CI.
    set_tests_properties(server_busy_guard_test PROPERTIES TIMEOUT 60)

    if (ENGINE_ENABLE_OPENMP)
    endif()



    # Requires the real (multi-GB) Parakeet-TDT weights, which aren't
    # downloaded as part of a normal checkout/CI build; the test itself
    # detects this and exits with kExitSkip (125) rather than failing.



    # Numerical parity dump tool (not registered as an add_test — it's a
    # manual/pre-release check, not a CI gate). Run it against a NeMo
    # reference dump via compare_parity.py; see tests/parakeet_tdt/parity/README.md.
    if (ENGINE_ENABLE_OPENMP)
    endif()
endif()

if (ENGINE_BUILD_EXAMPLES)
    add_executable(qwen3_tts_live examples/live_tts.cpp)
    target_link_libraries(qwen3_tts_live PRIVATE engine_runtime ggml)

    add_executable(ggml_simple_inference
        examples/ggml_simple_inference.cpp
    )

    target_link_libraries(ggml_simple_inference PRIVATE ggml)

    add_executable(model_spec_demo
        examples/model_spec_demo/model_spec_demo.cpp
    )
    target_link_libraries(model_spec_demo PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(model_spec_demo PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_executable(model_spec_download_demo
        examples/model_spec_demo/model_spec_download_demo.cpp
    )
    target_link_libraries(model_spec_download_demo PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(model_spec_download_demo PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
