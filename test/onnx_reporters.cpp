#include <rad/onnx/env.hpp>

#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

namespace onnx = rad::onnx;

class ONNXTestRunListener : public Catch::EventListenerBase
{
public:
    using Catch::EventListenerBase::EventListenerBase;

    void testRunStarting(Catch::TestRunInfo const&) override
    {
        if (!onnx::init_ort_api())
        {
            throw std::runtime_error{"error: onnx failed to initialize"};
        }
    }
};

CATCH_REGISTER_LISTENER(ONNXTestRunListener);
