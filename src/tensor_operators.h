#pragma once

// This file is part of the Marian toolkit.
// Marian is copyright (c) 2016 Marcin Junczys-Dowmunt.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cublas_v2.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>

#include "tensors/tensor_gpu.h"

namespace marian {

using namespace thrust::placeholders;
#define MAX_THREADS 512
#define MAX_BLOCKS 65535

class TensorGPU;

template <class Functor>
__global__ void gReduce(Functor functor,
                        float* out,
                        Shape outShape,
                        const float* in,
                        const Shape inShape,
                        const Shape full) {
  int length = full.elements();
  int dims[4];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length) {
      full.dims(index, dims);
      int outIndex = outShape.bindex(dims);
      int inIndex = inShape.bindex(dims);
      atomicAdd(out + outIndex,
                functor(in[inIndex]));
    }
  }
}

template <class Functor, class T1, class T2>
void Reduce(Functor functor,
         T1 out, T2 in) {

  auto full = out->shape();
  for(int i = 0; i < in->shape().size(); ++i)
    full.set(i, std::max(full[i], in->shape()[i]));

  int length = full.elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks  = std::min(MAX_BLOCKS, length / threads  + (length % threads != 0));

  out->set(0);
  gReduce<<<blocks, threads>>>(functor,
                               out->data(), out->shape(),
                               in->data(), in->shape(),
                               full);
  cudaStreamSynchronize(0);
}

template <class Functor>
__global__ void gReduce(Functor functor,
                        float* out,
                        Shape outShape,
                        const float* in1,
                        const Shape in1Shape,
                        const float* in2,
                        const Shape in2Shape,
                        const Shape full) {
  int length = full.elements();
  int dims[4];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length) {
      full.dims(index, dims);
      int outIndex = outShape.bindex(dims);
      int in1Index = in1Shape.bindex(dims);
      int in2Index = in2Shape.bindex(dims);
      atomicAdd(out + outIndex,
                functor(in1[in1Index], in2[in2Index]));
    }
  }
}

template <class Functor, class T1, class T2, class T3>
void Reduce(Functor functor,
            T1 out, T2 in1, T3 in2) {

  auto full = out->shape();
  for(int i = 0; i < in1->shape().size(); ++i)
    full.set(i, std::max(full[i], in1->shape()[i]));
  for(int i = 0; i < in2->shape().size(); ++i)
    full.set(i, std::max(full[i], in2->shape()[i]));

  int length = full.elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks  = std::min(MAX_BLOCKS, length / threads  + (length % threads != 0));

  out->set(0);
  gReduce<<<blocks, threads>>>(functor,
                               out->data(), out->shape(),
                               in1->data(), in1->shape(),
                               in2->data(), in2->shape(),
                               full);
  cudaStreamSynchronize(0);
}

template <class Functor>
__global__ void gReduce(Functor functor,
                        float* out,
                        Shape outShape,
                        const float* in1,
                        const Shape in1Shape,
                        const float* in2,
                        const Shape in2Shape,
                        const float* in3,
                        const Shape in3Shape,
                        const Shape full) {
  int length = full.elements();
  int dims[4];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length) {
      full.dims(index, dims);
      int outIndex = outShape.bindex(dims);
      int in1Index = in1Shape.bindex(dims);
      int in2Index = in2Shape.bindex(dims);
      int in3Index = in3Shape.bindex(dims);
      atomicAdd(out + outIndex,
                functor(in1[in1Index], in2[in2Index], in3[in3Index]));
    }
  }
}

template <class Functor, class T1, class T2, class T3, class T4>
void Reduce(Functor functor,
            T1 out, T2 in1, T3 in2, T4 in3) {

  auto full = out->shape();
  for(int i = 0; i < in1->shape().size(); ++i)
    full.set(i, std::max(full[i], in1->shape()[i]));
  for(int i = 0; i < in2->shape().size(); ++i)
    full.set(i, std::max(full[i], in2->shape()[i]));
  for(int i = 0; i < in3->shape().size(); ++i)
    full.set(i, std::max(full[i], in3->shape()[i]));

  int length = full.elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks  = std::min(MAX_BLOCKS, length / threads  + (length % threads != 0));

  out->set(0);
  gReduce<<<blocks, threads>>>(functor,
                               out->data(), out->shape(),
                               in1->data(), in1->shape(),
                               in2->data(), in2->shape(),
                               in3->data(), in3->shape(),
                               full);
  cudaStreamSynchronize(0);
}


template <class Functor>
__global__ void gElement(Functor functor,
                         float* out,
                         Shape outShape,
                         const float* in,
                         const Shape inShape,
                         bool broadcast) {
  int length = outShape.elements();
  int dims[4];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length) {
      int inIndex = index;
      if(broadcast) {
        outShape.dims(index, dims);
        inIndex = inShape.bindex(dims);
      }
      out[index] = functor(out[index], in[inIndex]);
    }
  }
}

template <class Functor, class T1, class T2>
void Element(Functor functor,
             T1 out, T2 in) {

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks  = std::min(MAX_BLOCKS, length / threads  + (length % threads != 0));

  gElement<<<blocks, threads>>>(functor,
                                out->data(), out->shape(),
                                in->data(), in->shape(),
                                out->shape() != in->shape());
  cudaStreamSynchronize(0);
}

template <class Functor>
__global__ void gElement(Functor functor,
                         float* out,
                         Shape outShape,
                         const float* in1,
                         const Shape inShape1,
                         const float* in2,
                         const Shape inShape2,
                         bool broadcast) {
  int length = outShape.elements();
  int dims[4];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length) {
      int inIndex1 = index;
      int inIndex2 = index;
      if(broadcast) {
        outShape.dims(index, dims);
        inIndex1 = inShape1.bindex(dims);
        inIndex2 = inShape2.bindex(dims);
      }
      out[index] = functor(out[index], in1[inIndex1], in2[inIndex2]);
    }
  }
}

template <class Functor, class T1, class T2, class T3>
void Element(Functor functor,
             T1 out, T2 in1, T3 in2) {

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks  = std::min(MAX_BLOCKS, length / threads  + (length % threads != 0));

  gElement<<<blocks, threads>>>(functor,
                                out->data(), out->shape(),
                                in1->data(), in1->shape(),
                                in2->data(), in2->shape(),
                                out->shape() != in1->shape() || out->shape() != in2->shape());

  cudaStreamSynchronize(0);
}

template <class Functor>
__global__ void gElement(Functor functor,
                         float* out,
                         Shape outShape,
                         const float* in1,
                         const Shape inShape1,
                         const float* in2,
                         const Shape inShape2,
                         const float* in3,
                         const Shape inShape3,
                         bool broadcast) {
  int length = outShape.elements();
  int dims[4];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length) {
      int inIndex1 = index;
      int inIndex2 = index;
      int inIndex3 = index;
      if(broadcast) {
        outShape.dims(index, dims);
        inIndex1 = inShape1.bindex(dims);
        inIndex2 = inShape2.bindex(dims);
        inIndex3 = inShape3.bindex(dims);
      }
      out[index] = functor(out[index], in1[inIndex1], in2[inIndex2], in3[inIndex3]);
    }
  }
}

template <class Functor, class T1, class T2, class T3, class T4>
void Element(Functor functor,
             T1 out, T2 in1, T3 in2, T4 in3) {

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks  = std::min(MAX_BLOCKS, length / threads  + (length % threads != 0));

  gElement<<<blocks, threads>>>(functor,
                                out->data(), out->shape(),
                                in1->data(), in1->shape(),
                                in2->data(), in2->shape(),
                                in3->data(), in3->shape(),
                                out->shape() != in1->shape()
                                || out->shape() != in2->shape()
                                || out->shape() != in3->shape());

  cudaStreamSynchronize(0);
}

template <class Functor>
__global__ void gElement(Functor functor,
                         float* out,
                         int length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length) {
      out[index] = functor(out[index]);
    }
  }
}

template <class Functor, class T1, class T2>
void Element(Functor functor, T1 out) {

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks  = std::min(MAX_BLOCKS, length / threads  + (length % threads != 0));

  gElement<<<blocks, threads>>>(functor, out->data(), length);
  cudaStreamSynchronize(0);
}


/**************** Pick ************************/

template <class Functor>
__global__ void gPick(Functor functor,
                      float* out,
                      Shape outShape,
                      const size_t* d_pick) {
  int length = outShape.elements();
  int dims[4];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length) {
      outShape.dims(index, dims);
      int row = dims[0];
      int col = dims[1];
      float picked = col == (int)d_pick[row];
      out[index] = functor(out[index], picked);
    }
  }
}

template <class Functor, class T1>
void Pick(Functor functor, T1 out, const DeviceVector<size_t>& picks) {

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks  = std::min(MAX_BLOCKS, length / threads  + (length % threads != 0));

  gPick<<<blocks, threads>>>(functor, out->data(), out->shape(),
                             thrust::raw_pointer_cast(picks.data()));
  cudaStreamSynchronize(0);
}


template <class Functor>
__global__ void gPick(Functor functor,
                      float* out,
                      Shape outShape,
                      const float* in,
                      const size_t* d_pick) {
  int length = outShape.elements();
  int dims[4];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length) {
      outShape.dims(index, dims);
      int row = dims[0];
      int col = dims[1];
      float picked = col == (int)d_pick[row];
      out[index] = functor(out[index], in[index], picked);
    }
  }
}

template <class Functor, class T1, class T2>
void Pick(Functor functor, T1 out, const T2 in, const DeviceVector<size_t>& picks) {

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks  = std::min(MAX_BLOCKS, length / threads  + (length % threads != 0));

  gPick<<<blocks, threads>>>(functor, out->data(), out->shape(),
                             in->data(),
                             thrust::raw_pointer_cast(picks.data()));
  cudaStreamSynchronize(0);
}

template <class Functor>
__global__ void gPickReduce(Functor functor,
                      float* out,
                      Shape outShape,
                      const float* in,
                      const Shape inShape,
                      const size_t* d_pick) {
  int length = inShape.elements();
  int dims[4];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length) {
      inShape.dims(index, dims);
      int row = dims[0];
      int col = dims[1];
      int outIndex = outShape.bindex(dims);
      float picked = col == (int)d_pick[row];
      float result = functor(in[index], picked);

      if(result)
        atomicAdd(out + outIndex, result);
    }
  }
}

template <class Functor, class T1, class T2>
void PickReduce(Functor functor, T1 out, const T2 in, const DeviceVector<size_t>& picks) {

  int length = in->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks  = std::min(MAX_BLOCKS, length / threads  + (length % threads != 0));

  out->set(0);
  gPickReduce<<<blocks, threads>>>(functor, out->data(), out->shape(),
                             in->data(), in->shape(),
                             thrust::raw_pointer_cast(picks.data()));
  cudaStreamSynchronize(0);
}


template <class Functor>
__global__ void gPick(Functor functor,
                      float* out,
                      Shape outShape,
                      const float* in1,
                      const float* in2,
                      const size_t* d_pick) {
  int length = outShape.elements();
  int dims[4];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length) {
      outShape.dims(index, dims);
      int row = dims[0];
      int col = dims[1];
      float picked = col == (int)d_pick[row];
      out[index] = functor(out[index], in1[index], in2[index], picked);
    }
  }
}

template <class Functor, class T1, class T2, class T3>
void Pick(Functor functor, T1 out, const T2 in1, const T3 in2, const DeviceVector<size_t>& picks) {

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks  = std::min(MAX_BLOCKS, length / threads  + (length % threads != 0));

  gPick<<<blocks, threads>>>(functor, out->data(), out->shape(),
                             in1->data(), in2->data(),
                             thrust::raw_pointer_cast(picks.data()));
  cudaStreamSynchronize(0);
}

void ClipNorm(Tensor out, float threshold);

void SubtractMax(Tensor out, Tensor in);

void Softmax(Tensor out, Tensor in);

void SoftmaxGrad(Tensor grad, Tensor adj, Tensor val);
void LogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val);

void CudnnSoftmax(Tensor out, Tensor in);
void CudnnSoftmaxGrad(Tensor grad, Tensor adj, Tensor val);

void CudnnLogSoftmax(Tensor out, Tensor in);
void CudnnLogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val);

void Argmax(Tensor Out, const Tensor In);

void Prod(cublasHandle_t handle, Tensor C, const Tensor A, const Tensor B,
             bool transA, bool transB, Float beta);

void Prod(Tensor C, const Tensor A, const Tensor B,
             bool transA, bool transB, Float beta = 0);

void CopyRowsByIndex(Tensor out, const Tensor in,
                     thrust::pair<size_t, size_t>* ipair, size_t length);

void CopyRows(Tensor out, const Tensor in, const DeviceVector<size_t>& indeces);

void PasteRows(Tensor out, const Tensor in, const DeviceVector<size_t>& indeces);

void CudnnDropoutPrepare(Tensor in, float p,
                         cudnnDropoutDescriptor_t* dropDesc,
                         void** space, size_t* spaceSize,
                         void** states, size_t seed);

void CudnnDropoutDestroy(cudnnDropoutDescriptor_t dropDesc,
                         void* space, void* states);

void CudnnDropoutForward(cudnnDropoutDescriptor_t dropoutDesc,
                  void* space, size_t spaceSize,
                  Tensor out, Tensor in);

void CudnnDropoutBackward(cudnnDropoutDescriptor_t dropoutDesc,
                          void* space, size_t spaceSize,
                          Tensor out, Tensor in);

void Transpose(Tensor out, const Tensor in);

void Concatenate(Tensor out, const std::vector<Tensor>& inputs, int ax);

void Deconcatenate(std::vector<Tensor>& outputs, const Tensor in, int ax);

void GRUFastForward(Tensor out, const std::vector<Tensor>& inputs, bool final = false);

void GRUFastBackward(std::vector<Tensor>& outputs,
                     const std::vector<Tensor>& inputs,
                     const Tensor adj, bool final = false);

}
