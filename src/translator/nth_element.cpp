/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "translator/nth_element.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>

#include <iostream>
#include <fstream>
#include <map>
#include <boost/algorithm/string.hpp>
// #include <math.h>

namespace marian {

class NthElementCPU {
  std::vector<int> h_res_idx;
  std::vector<float> h_res;
  //size_t lastN_;

public:
  NthElementCPU() {}
  NthElementCPU(const NthElementCPU& copy) = delete;


public:
void getNBestList(Tensor scores, // [dimBatch, 1, beamSize, dimVocab or dimShortlist]
                    size_t N,
                    std::vector<float>& outPathScores,
                    std::vector<unsigned>& outKeys,
                    const bool isFirst,
                    // size_t t,
                    // float beamSizeDivideBy,
                    // size_t beamSizeDivideMin,
                    std::vector<std::vector<int>>& trieVocabIdxs,
                    float * cpumem) {
    
    // vocabMap used for debugging
    /*
    std::map<int, std::string> vocabMap;
    std::string delimiter = ": ";
    std::ifstream input( "/home/patrick/Desktop/marian-dev/examples/trieme_new/model/vocab.deen.yml" );
    int count = 0;
    for( std::string line; getline( input, line ); ) {
      boost::trim_right(line);
      std::string token = line.substr(0, line.find(delimiter));
      // std::cout << token << " is " << count << ", ";
      vocabMap[count] = token;
      ++count;
    } */
    const auto vocabSize = scores->shape()[-1];
    const auto inputN    = scores->shape()[-2];
    const auto dimBatch  = scores->shape()[-4];
    // size_t dynamic_N = std::max((size_t)(float(N) / pow(beamSizeDivideBy, t)), beamSizeDivideMin);

    float* scoresData = nullptr;
    #ifdef CUDA_FOUND
    if (scores->getDeviceId().type == DeviceType::gpu) {
      // Finish the asynchronous GPU copy
      syncStreamZero();
      scoresData = cpumem;
    } else {
      ABORT_IF(inputN != (isFirst ? 1 : N), "Input tensor has wrong beam dim??"); // @TODO: Remove isFirst argument altogether
      scoresData  = scores->data();
    }
    #else
    ABORT_IF(inputN != (isFirst ? 1 : N), "Input tensor has wrong beam dim??"); // @TODO: Remove isFirst argument altogether
    scoresData  = scores->data();
    #endif

    h_res.clear();
    h_res_idx.clear();
  
    size_t batchOffset = inputN * vocabSize; // offset for each beam in float* scoresData
    for(size_t batchIdx = 0; batchIdx < dimBatch; ++batchIdx) {
       // ensure that there is a continuation. if there is no continuation,
       // scores matrix do not have scores for continuations of the hypoetheses
       // correspond to beam_search.h line 532. could be redundant.
      if (trieVocabIdxs[batchIdx].size() > 0) {
        std::vector<int> idxs = trieVocabIdxs[batchIdx]; // continuations (with offset) for all hyps
        std::partial_sort(
          idxs.begin(),
          // idxs.begin() + std::min(dynamic_N, idxs.size()), // only sort min(max_beam_size, num_of_continuations) 
          idxs.begin() + std::min(N, idxs.size()), // only sort min(max_beam_size, num_of_continuations) 
          idxs.end(),
          [&](int a, int b) {return scoresData[a] > scoresData[b]; } // compare by score. note a and b are with offset
        );
        // for(int temp = 0; temp < std::min(dynamic_N, idxs.size()); ++temp) {
        for(int temp = 0; temp < std::min(N, idxs.size()); ++temp) {
          int idx = idxs[temp];
          // move selected idxs to return vector.
          // note idx is with offset for hypotheses but not batch, so add batch offset too
          h_res_idx.push_back((int)(idx + batchIdx * batchOffset)); 
          // scores do not need offset because the pointer gets advanced each time as below.
          h_res.push_back(scoresData[idx]);
        }
        scoresData += batchOffset; //advance score pointer to start of next batch
      }
    }
    getPairs(/*cumulativeBeamSizes.back(),*/ outKeys, outPathScores);
  }

private:
  void getPairs(/*size_t number,*/
                std::vector<unsigned>& outKeys,
                std::vector<float>& outValues) {
    std::copy(h_res_idx.begin(), h_res_idx.end(), std::back_inserter(outKeys));
    std::copy(h_res    .begin(), h_res    .end(), std::back_inserter(outValues));
    
    //lastN_ = number;
  }

  //void getValueByKey(std::vector<float>& out, float* d_in) {
  //  for(size_t i = 0; i < lastN_; ++i) {
  //    out[i] = d_in[h_res_idx[i]];
  //  }
  //}
};

#ifdef CUDA_FOUND
GetNBestListFn createGetNBestListGPUFn(size_t beamSize, size_t dimBatch, DeviceId deviceId); // in .cu file
#endif

// factory function
// Returns a lambda with the same signature as the getNBestList() function.
GetNBestListFn createGetNBestListFn(size_t beamSize, size_t dimBatch, DeviceId deviceId, bool triePrune) {
#ifdef CUDA_FOUND
  if(deviceId.type == DeviceType::gpu && !triePrune)
    return createGetNBestListGPUFn(beamSize, dimBatch, deviceId);
#else
  deviceId; beamSize; dimBatch; // (unused)
#endif
  auto nth = New<NthElementCPU>();
  return [nth](Tensor logProbs, size_t N, std::vector<float>& outCosts, std::vector<unsigned>& outKeys, const bool isFirst, 
    /* size_t t, float beamSizeDivideBy, size_t beamSizeDivideMin,*/ std::vector<std::vector<int>>& trieVocabIdxs, float * cputensor=nullptr) {
    return nth->getNBestList(logProbs, N, outCosts, outKeys, isFirst, /* t, beamSizeDivideBy, beamSizeDivideMin, */ trieVocabIdxs, cputensor);
  };
}

}  // namespace marian
