#pragma once

#if defined(RAD_ONNX_ENABLED)
#    define ORT_API_MANUAL_INIT
#    include <onnxruntime_cxx_api.h>
#endif
