#include "Simd/SimdLib.h"
#include "Simd/SimdSynetConvolution32f.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace
{
    const size_t HeaderSize = 15;
    const size_t FloatBudget = 8 * 1024 * 1024 / sizeof(float);
    const uint64_t MacBudget = 2097152;

    struct Input
    {
        const uint8_t* data;
        size_t size;

        size_t Value(size_t offset) const
        {
            return uint8_t(data[offset] - uint8_t('0'));
        }
    };

    struct Release
    {
        void operator()(void* context) const
        {
            if (context)
                SimdRelease(context);
        }
    };

    typedef std::unique_ptr<void, Release> Context;

    bool Add(size_t count, size_t& total)
    {
        if (count > FloatBudget - total)
            return false;
        total += count;
        return true;
    }

    bool Axis(size_t source, size_t kernel, size_t dilation, size_t stride,
        size_t before, size_t after, size_t& destination)
    {
        if (!source || source > 32 || !kernel || kernel > 7 || !dilation || dilation > 2 ||
            !stride || stride > 3 || before > 3 || after > 3)
            return false;
        const int64_t effective = int64_t(dilation) * (int64_t(kernel) - 1) + 1;
        const int64_t available = int64_t(source) + int64_t(before) + int64_t(after) - effective;
        if (available < 0)
            return false;
        const int64_t output = available / int64_t(stride) + 1;
        if (output < 1 || output > 32)
            return false;
        destination = size_t(output);
        return true;
    }

    bool Shape(const Input& input, SimdConvolutionParameters& conv, size_t& batch)
    {
        const size_t family = (size_t(input.data[0]) + 256 - size_t('A')) % 8;
        batch = 1 + input.Value(1) % 2;
        conv.srcC = 1 + (31 + input.Value(4)) % 64;
        conv.dstC = 1 + (31 + input.Value(5)) % 64;
        conv.srcT = conv.dstT = SimdTensorData32f;
        conv.srcF = conv.dstF = SimdTensorFormatNhwc;
        conv.group = 1;
        conv.kernelY = conv.kernelX = 3;
        conv.strideY = conv.strideX = 1;
        conv.dilationY = conv.dilationX = 1;
        conv.activation = input.Value(12) % 2 ? SimdConvolutionActivationRelu : SimdConvolutionActivationIdentity;

        if (family == 0)
        {
            conv.srcF = conv.dstF = SimdTensorFormatNchw;
            conv.srcC = 3 - input.Value(4) % 3;
            conv.srcH = 7 + input.Value(2) % 4;
            conv.srcW = 7 + input.Value(3) % 4;
            conv.strideY = conv.strideX = 2 - input.Value(6) % 2;
            conv.padY = conv.padX = conv.padH = conv.padW = input.Value(8) % 2;
        }
        else if (family == 1 || family == 2)
        {
            conv.kernelY = conv.kernelX = family == 1 ? 3 : 7;
            conv.srcH = conv.kernelY + input.Value(2) % (33 - conv.kernelY);
            conv.srcW = conv.kernelX + input.Value(3) % (33 - conv.kernelX);
            conv.dstC = conv.group = conv.srcC;
            conv.strideY = 1 + input.Value(6) % 3;
            conv.strideX = 1 + input.Value(7) % 3;
            conv.padY = input.Value(8) % (conv.kernelY / 2 + 1);
            conv.padX = input.Value(9) % (conv.kernelX / 2 + 1);
            conv.padH = input.Value(10) % (conv.kernelY / 2 + 1);
            conv.padW = input.Value(11) % (conv.kernelX / 2 + 1);
        }
        else if (family == 3)
        {
            conv.kernelY = conv.kernelX = 1;
            conv.srcH = 1 + input.Value(2) % 32;
            conv.srcW = 1 + input.Value(3) % 32;
        }
        else if (family == 4)
        {
            conv.srcF = conv.dstF = SimdTensorFormatNchw;
            conv.group = size_t(2) << (input.Value(7) % 3);
            conv.srcC = conv.group * (1 + (7 + input.Value(4)) % (64 / conv.group));
            conv.dstC = conv.group * (1 + (7 + input.Value(5)) % (64 / conv.group));
            conv.srcH = 13 + input.Value(2) % 20;
            conv.srcW = 13 + input.Value(3) % 20;
            conv.kernelY = conv.kernelX = 3 + 2 * (input.Value(7) / 3 % 3);
            conv.dilationY = 2;
            conv.dilationX = 1 + input.Value(6) % 2;
            conv.strideY = 1 + input.Value(6) / 2 % 3;
            conv.strideX = 1 + input.Value(7) / 9 % 3;
            conv.padY = input.Value(8) % (conv.kernelY / 2 + 1);
            conv.padX = input.Value(9) % (conv.kernelX / 2 + 1);
            conv.padH = input.Value(10) % (conv.kernelY / 2 + 1);
            conv.padW = input.Value(11) % (conv.kernelX / 2 + 1);
        }
        else
        {
            const size_t heights[] = { 6, 9, 12 };
            const size_t widths[] = { 7, 9, 12 };
            conv.srcH = heights[family - 5] + input.Value(2) % 2;
            conv.srcW = widths[family - 5] + input.Value(3) % 2;
            conv.srcC = 17 + (15 + input.Value(4)) % 48;
            conv.dstC = 17 + (15 + input.Value(5)) % 48;
            conv.padY = conv.padX = conv.padH = conv.padW = 1 - input.Value(8) % 2;
        }

        if (!conv.srcC || conv.srcC > 64 || !conv.dstC || conv.dstC > 64 ||
            !conv.group || conv.srcC % conv.group || conv.dstC % conv.group)
            return false;
        return Axis(conv.srcH, conv.kernelY, conv.dilationY, conv.strideY, conv.padY, conv.padH, conv.dstH) &&
            Axis(conv.srcW, conv.kernelX, conv.dilationX, conv.strideX, conv.padX, conv.padW, conv.dstW);
    }

    void Fill(std::vector<float>& values, const Input& input, size_t& offset)
    {
        const size_t payload = input.size - HeaderSize;
        for (size_t index = 0; index < values.size(); ++index)
        {
            values[index] = payload ? float(input.data[HeaderSize + offset % payload]) / 127.5f - 1.0f : 0.0f;
            ++offset;
        }
    }

    bool Fits(void* first, void* second, const size_t external[2], size_t caller, size_t reserve)
    {
        size_t total = caller;
        return Add(SimdSynetConvolution32fInternalBufferSize(first), total) &&
            Add(SimdSynetConvolution32fInternalBufferSize(second), total) &&
            Add(external[0], total) && Add(external[1], total) && Add(reserve, total);
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    if (size < HeaderSize)
        return 0;
    const Input input = { data, size };
    SimdConvolutionParameters conv = {};
    size_t batch = 0;
    if (!Shape(input, conv, batch))
        return 0;
    const uint64_t macs = uint64_t(batch) * conv.dstH * conv.dstW * conv.dstC *
        (conv.srcC / conv.group) * conv.kernelY * conv.kernelX;
    if (macs > MacBudget || macs * 2 > 4194304)
        return 0;

    const size_t sourceCount = batch * conv.srcC * conv.srcH * conv.srcW;
    const size_t weightCount = conv.kernelY * conv.kernelX * conv.srcC * conv.dstC / conv.group;
    const size_t destinationCount = batch * conv.dstC * conv.dstH * conv.dstW;
    const size_t parameterCount = std::max(size_t(2), conv.dstC);
    size_t caller = 0;
    if (!Add(sourceCount, caller) || !Add(weightCount, caller) || !Add(destinationCount, caller) ||
        !Add(conv.dstC, caller) || !Add(parameterCount, caller))
        return 0;

    std::vector<float> source(sourceCount), weight(weightCount), destination(destinationCount);
    std::vector<float> bias(conv.dstC), parameters(parameterCount), scratch;
    size_t offset = 0;
    Fill(source, input, offset);
    Fill(weight, input, offset);
    Fill(destination, input, offset);
    Fill(bias, input, offset);
    Fill(parameters, input, offset);

    Context base(Simd::Base::SynetConvolution32fInit(batch, &conv));
    Context dispatched(SimdSynetConvolution32fInit(batch, &conv));
    if (!base || !dispatched)
        return 0;
    void* contexts[] = { base.get(), dispatched.get() };
    const size_t external[] = { SimdSynetConvolution32fExternalBufferSize(contexts[0]),
        SimdSynetConvolution32fExternalBufferSize(contexts[1]) };
    const size_t parameterReserve = 2 * (std::max(weightCount, size_t(36) * conv.srcC * conv.dstC) + 256);
    if (!Fits(contexts[0], contexts[1], external, caller, parameterReserve))
        return 0;
    const float* biasData = input.Value(14) % 2 ? NULL : bias.data();
    for (size_t index = 0; index < 2; ++index)
    {
        SimdBool internal = SimdFalse;
        SimdSynetConvolution32fSetParams(contexts[index], weight.data(), &internal, biasData, parameters.data());
        (void)SimdSynetConvolution32fInfo(contexts[index]);
    }
    if (!Fits(contexts[0], contexts[1], external, caller, 0))
        return 0;

    const size_t callerIndex = input.Value(13) % 2;
    scratch.resize(std::max(size_t(1), external[callerIndex]), 0.0f);
    SimdSynetConvolution32fForward(contexts[callerIndex], source.data(), scratch.data(), destination.data());
    if (!Fits(contexts[0], contexts[1], external, caller, 0))
        return 0;
    SimdSynetConvolution32fForward(contexts[1 - callerIndex], source.data(), NULL, destination.data());
    return 0;
}
