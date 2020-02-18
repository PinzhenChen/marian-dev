/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include "tensors/tensor.h"
#include <vector>

namespace marian {

typedef std::function<void(Tensor logProbs,
                           size_t N,
                           std::vector<float>& outCosts,
                           std::vector<unsigned>& outKeys,
                           const bool isFirst,
                           std::vector<std::vector<int>> trieVocabIdxs)> GetNBestListFn;

GetNBestListFn createGetNBestListFn(size_t beamSize, size_t dimBatch, DeviceId deviceId);
void gBumpScores(std::vector<uint32_t>& ids, float * in_, float bumpVal);
}  // namespace marian
