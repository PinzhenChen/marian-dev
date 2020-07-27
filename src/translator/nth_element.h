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
                        //    size_t t,
                        //    float beamSizeDivideBy,
                        //    size_t beamSizeDivideMin,
                           std::vector<std::vector<int>>& trieVocabIdxs,
                           float * cputensor)> GetNBestListFn;

GetNBestListFn createGetNBestListFn(size_t beamSize, size_t dimBatch, DeviceId deviceId, bool triePrune=false);

float * getPinnedMemory(size_t size);
void freePinnedMemory(float * mem);
void copyTensorToCpuAsync(float * cpumem, float * gpumem, size_t size);
void syncStreamZero(); // Since we use the async API to copy in order to overlay some operations

}  // namespace marian
