/**
 * @file pipeline.cpp
 * @author joker.mao (joker_mao@163.com)
 * @brief
 * @version 0.1
 * @date 2023-07-27
 *
 * Copyright (c) of ADAS_EYES 2023
 *
 */
#include "modules/modules.h"
#include "common/pipeline.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

namespace
{

using ModuleTiming = std::pair<std::string, double>;

int ModuleProfileReportEvery()
{
    static const int report_every = [] {
        const char *value = std::getenv("CYPERSTEREO_HDRISP_PROFILE_MODULES");
        if (value == nullptr)
            return 0;

        char *end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0')
            return 100;
        if (parsed <= 0)
            return 0;
        return static_cast<int>((std::min)(parsed, 1000000L));
    }();
    return report_every;
}

double Percentile(std::vector<double> values, double q)
{
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const double position = q * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = (std::min)(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

void PrintStats(const char *name, const std::vector<double> &samples)
{
    if (samples.empty())
        return;
    const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    std::printf("[hdr-isp-module] %-12s avg=%8.3f ms  p50=%8.3f ms  "
                "p95=%8.3f ms\n",
                name, sum / samples.size(), Percentile(samples, 0.50),
                Percentile(samples, 0.95));
}

void AddModuleProfileFrame(const std::vector<ModuleTiming> &timings)
{
    static std::mutex mutex;
    static std::map<std::string, std::vector<double>> samples;
    static std::vector<std::string> module_order;
    static std::vector<double> totals;
    static int frame_count = 0;

    std::lock_guard<std::mutex> lock(mutex);
    double total = 0.0;
    for (const auto &timing : timings)
    {
        auto inserted = samples.emplace(timing.first, std::vector<double>{});
        if (inserted.second)
            module_order.push_back(timing.first);
        inserted.first->second.push_back(timing.second);
        total += timing.second;
    }
    totals.push_back(total);

    const int report_every = ModuleProfileReportEvery();
    if (++frame_count < report_every)
        return;

    std::printf("[hdr-isp-modules] frames=%d\n", frame_count);
    for (const auto &name : module_order)
        PrintStats(name.c_str(), samples[name]);
    PrintStats("modules-total", totals);
    std::fflush(stdout);

    for (auto &entry : samples)
        entry.second.clear();
    totals.clear();
    frame_count = 0;
}

} // namespace

void IspInit()
{
    RegisterUnpackMod();
    RegisterDePwlMod();
    RegisterBlcMod();
    RegisterDemoasicMod();
    RegisterCcmMod();
    RegisterYGammaMod();
    RegisterWbGaincMod();
    RegisterLtmMod();
    RegisterRgbGammaMod();
    RegisterYuv2RgbMod();
    RegisterRgb2YuvMod();
    RegisterSaturationMod();
    RegisterContrastMod();
    RegisterContrastMod();
    RegisterSharpenMod();
    RegisterLscMod();
    RegisterDpcMod();
    RegisterCnsMod();
    RegisterBayerNrMod();
    RegisterLumaNrMod();
    RegisterChromaNrMod();
    RegisterFalseColorMod();
}

IspPipeline::IspPipeline()
{
    pipe_.clear();
    IspInit();
    ShowAllIspModules();
}
IspPipeline::~IspPipeline()
{
    pipe_.clear();
}

IspPipeline::IspPipeline(std::list<std::string> pipeline)
{
    IspPipeline();
    MakePipe(pipeline);
}

int IspPipeline::MakePipe(const std::list<std::string> &pipeline_str)
{
    IspModule mod;
    IspModule last_mod;
    for (auto item : pipeline_str)
    {
        if (0 == GetIspModuleFromName(item, mod))
        {
            if (pipe_.size() > 0)
            {
                if ((mod.in_type != last_mod.out_type) || (mod.in_domain != last_mod.out_domain))
                {
                    if ((mod.in_domain != last_mod.out_domain))
                        LOG(ERROR) << "mod " << mod.name << " domain is not equal wait " << last_mod.name;
                    if (mod.in_type != last_mod.out_type)
                        LOG(ERROR) << "mod " << mod.name << " in bit is not equal wait " << last_mod.name;
                    is_pipe_vaild_ = false;
                    return -1;
                }
            }
            pipe_.push_back(mod);
            last_mod = mod;
        }
        else
        {
            LOG(WARNING) << item << " find failed";
        }
    }
    is_pipe_vaild_ = true;
    return 0;
}

int IspPipeline::RunPipe(Frame *frame, const IspPrms *prms)
{
    if (!is_pipe_vaild_)
    {
        LOG(ERROR) << "pipeline is not vailed..";
        return -1;
    }
    const bool profile_modules = ModuleProfileReportEvery() > 0;
    std::vector<ModuleTiming> timings;
    if (profile_modules)
        timings.reserve(pipe_.size());

    LOG(INFO) << "============= user pipeline running ==============";
    for (const auto &isp_mod : pipe_)
    {
        const auto start = std::chrono::steady_clock::now();
        if (isp_mod.run_function(frame, prms) != 0)
        {
            LOG(ERROR) << "pipeline run failed, mod " << isp_mod.name;
            return -1;
        }
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        LOG(INFO) << "mod " << isp_mod.name << "\t time: " << elapsed_ms << "ms";
        if (profile_modules)
            timings.emplace_back(isp_mod.name, elapsed_ms);
    }
    LOG(INFO) << "============= user pipeline running end ==============";
    if (profile_modules)
        AddModuleProfileFrame(timings);
    return 0;
}

int IspPipeline::PrintPipe()
{
    if (!is_pipe_vaild_)
    {
        LOG(ERROR) << "pipeline is not vailed..";
        return -1;
    }
    int index = 0;
    LOG(INFO) << "============= user pipeline print start ==============";
    for (auto isp_mod : pipe_)
    {
        LOG(INFO) << "mod[" << index++ << "] -> " << isp_mod.name;
    }
    LOG(INFO) << "============= user pipeline print end ==============";
    return 0;
}
