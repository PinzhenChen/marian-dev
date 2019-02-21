#include "training/graph_group_async.h"
#include "data/corpus_base.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"

namespace marian {

AsyncGraphGroup::AsyncGraphGroup(Ptr<Options> options, Ptr<IMPIWrapper> mpi)
    : GraphGroup(options, Config::getDevices(options)),
      shardSync_(devices_.size()),
      optimizerDelay_((size_t)options_->get<double>("optimizer-delay")) {
  ABORT_IF(mpi->numMPIProcesses() != 1, "AsyncGraphGroup presently does not support multiple MPI processes");
  ABORT_IF((double)optimizerDelay_ != options_->get<double>("optimizer-delay"), "AsyncGraphGroup presently does not implement fractional values for --optimizer-delay");
  pool_.reset(new ThreadPool(devices_.size(), devices_.size()));

  for(auto device : devices_) {
    auto graph = New<ExpressionGraph>();
    graph->setDevice(device);

    auto precisions = options_->get<std::vector<std::string>>("precision");
    graph->setParameterType(typeFromString(precisions[0]));

    graph->getBackend()->setClip(options_->get<float>("clip-gemm"));
    graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
    graphs_.push_back(graph);
    optimizerShards_.push_back(Optimizer(options_));
    models_.push_back(models::from_options(options_, models::usage::training));
  }
}

void AsyncGraphGroup::setScheduler(Ptr<Scheduler> scheduler) {
  scheduler_ = scheduler;
  // optimizer has to be registered last to see changes of learning rate
  scheduler_->registerTrainingObserver(scheduler_);
  for(auto opt : optimizerShards_)
    scheduler_->registerTrainingObserver(opt);
}

void AsyncGraphGroup::fetchParams(Tensor oldParams,
                                  const std::vector<Tensor>& params,
                                  int /*device_id*/) {
  // @TODO read guard on parameters
  int pos = 0;
  auto fetch = [&](int idx, int pos) {
    // individual mutex per-shard
    std::lock_guard<std::mutex> guard(shardSync_[idx]);
    oldParams->subtensor((int)pos, (int)params[idx]->size())->copyFrom(params[idx]);
  };

  std::vector<std::thread> threads;
  for(int idx = 0; idx < devices_.size(); idx++) {
    threads.emplace_back(std::thread(fetch, idx, pos));
    pos += shardSize_;
  }
  for(auto&& t : threads)
    t.join();
}

void AsyncGraphGroup::pushGradients(Tensor newGrads,
                                    int /*device_id*/) {
  std::vector<std::thread> threads;
  int pos = 0;
  for(int idx = 0; idx < devices_.size(); idx++) {
    auto push = [&](int idx, int pos) {
      // individual mutex per-shard
      std::lock_guard<std::mutex> guard(shardSync_[idx]);
      grads_[idx]->copyFrom(newGrads->subtensor(pos, (int)grads_[idx]->size()));
      optimizerShards_[idx]->update(params_[idx], grads_[idx]);
    };

    threads.emplace_back(std::thread(push, idx, pos));
    pos += shardSize_;
  }
  for(auto&& t : threads)
    t.join();
}

void AsyncGraphGroup::init(Ptr<data::Batch> batch) {
  // initialize the parameters
  {
    ThreadPool pool(graphs_.size(), graphs_.size());
    for(size_t i = 0; i < graphs_.size(); ++i) {
      auto init = [&](size_t i) {
        models_[i]->build(graphs_[i], batch);
        graphs_[i]->forward();
      };
      pool.enqueue(init, i);
    }
  }

  if(params_.empty()) {
    int totalSize = (int)graphs_[0]->params()->vals()->size();
    shardSize_ = (int)ceil(totalSize / (float)devices_.size());

    int pos = 0;
    // parameter sharding
    for(auto graph : graphs_) {
      int __size__ = std::min(shardSize_, totalSize);
      totalSize -= __size__;

      Tensor param;
      Ptr<TensorAllocator> allocator
          = New<TensorAllocator>(graph->getBackend());
      allocator->reserveExact(__size__ * sizeOf(graph->getParameterType()));
      allocator->allocate(param, {1, __size__}, graph->getParameterType());
      paramsAlloc_.push_back(allocator);

      param->copyFrom(graphs_[0]->params()->vals()->subtensor(pos, __size__));
      params_.push_back(param);

      pos += __size__;
    }
  }
  if(grads_.empty()) {
    int totalSize = (int)graphs_[0]->params()->vals()->size();

    for(auto graph : graphs_) {
      int __size__ = std::min(shardSize_, totalSize);
      totalSize -= __size__;
      Tensor grad;
      Ptr<TensorAllocator> allocator
          = New<TensorAllocator>(graph->getBackend());

      allocator->reserveExact(__size__ * sizeOf(graph->getParameterType()));
      allocator->allocate(grad, {1, __size__}, graph->getParameterType());
      grad->set(0.f);

      gradsAlloc_.push_back(allocator);
      grads_.push_back(grad);
    }
  }

  // Initialize optimizers with empty gradient
  for(int i = 0; i < params_.size(); ++i)
    optimizerShards_[i]->update(params_[i], grads_[i]);
}

void AsyncGraphGroup::execute(Ptr<data::Batch> batch) {
  if(first_) {
    init(batch);
    first_ = false;
  }

  size_t workers = devices_.size();
  auto task = [this, &threadCount, workers](Ptr<data::Batch> batch) {
    // assign thread id safely via atomic increment  
    static std::atomic<int> threadCount{0};
    thread_local int tid = -1;
    if(tid == -1)
      tid = threadCount++;

    thread_local size_t t = 0;
    thread_local size_t num_seen_words = 0;
    thread_local size_t num_seen_sentences = 0;
    thread_local StaticLoss loss;

    thread_local Tensor accGradients;
    thread_local Ptr<TensorAllocator> accAlloc;

    ABORT_IF(costScale_ ,"Cost-scaling not implemented for AsyncSGD");

    auto graph = graphs_[tid];
    Ptr<RationalLoss> dynamicLoss = models_[tid]->build(graph, batch);
    if(costScaleFactor_ != 1.f) {
      // it's ok to go out of scope, this will still insert the new top node into the graph
      auto costNode = dynamicLoss->loss() * costScaleFactor_;
    }

    if(t % optimizerDelay_ == 0) {
      fetchParams(graph->params()->vals(), params_, tid);
    }

    graph->forward();
    loss += *dynamicLoss; // does not add scaledLoss but original loss
    graph->backward();

    Tensor gradients;
    if(optimizerDelay_ > 1) {
      if(t == 0) {
        accAlloc = New<TensorAllocator>(graph->getBackend());
        accAlloc->reserveExact(graph->params()->grads()->memory()->size());
        accAlloc->allocate(accGradients, graph->params()->grads()->shape(), graph->getParameterType());
        accGradients->set(0);
      }

      using namespace functional;
      Element(_1 += _2, accGradients, graph->params()->grads());
      gradients = accGradients;

      // Keep track of how many words we've calculated the error from
      num_seen_words += batch->words();
      num_seen_sentences += batch->size();
    } else {
      gradients = graph->params()->grads();
    }

    t++;

    if(t % optimizerDelay_ == 0) {
      pushGradients(gradients, tid);
      // Reset the counter of seen target words after gradient update
      if(optimizerDelay_ > 1)
        gradients->set(0);
    }

    if(t % optimizerDelay_ == 0 && scheduler_) {
      std::unique_lock<std::mutex> lock(schedulerMutex_);

      // Wait until the thread that wants to do validation is finished.
      pool_->wait_for_one(lock);

      if(optimizerDelay_ > 1) {
        std::vector<size_t> fakeLength = {1, 1};
        std::vector<Ptr<Vocab>> vocabs;
        auto fb = data::CorpusBatch::fakeBatch(fakeLength, vocabs, num_seen_sentences, NULL);
        fb->front()->setWords(num_seen_words);

        scheduler_->update(loss, fb);

        num_seen_words = 0;
        num_seen_sentences = 0;
      } else {
        scheduler_->update(loss, batch);
      }

      loss.reset();

      if(scheduler_->saving() || scheduler_->validating()) {
        // Wait with validation or saving until all other threads are done with
        // update.
        // We want to reuse the graphs for validation, so they need to be in
        // a safe state.
        pool_->wait_for_others(lock);

        if(scheduler_->validating())
          scheduler_->validate(graphs_);

        if(scheduler_->saving())
          save(); // since we have waited above we can just call the generic save

        // Validation or saving is done, tell other threads to continue work.
        pool_->notify_others();
      }
    }
  };

  pool_->enqueue(task, batch);
}

void AsyncGraphGroup::load() {
  auto scatterFn = [&](const io::Item& data, const OptimizerBase::ScatterStateSetFunc& setFn) {
    size_t dataSize = data.size();
    size_t numLocalDevices = graphs_.size();
    size_t shardSize = (dataSize + numLocalDevices - 1) / numLocalDevices;// (size_t)(ceil(dataSize / (float)numLocalDevices));
    for (size_t i = 0; i < numLocalDevices; i++) {
      size_t begin = i * shardSize;
      size_t end = std::min(begin + shardSize, dataSize);
      setFn(i, data.bytes.data() + begin, data.bytes.data() + end);
    }
  };

  // This function loads the main parameters in the graphs.
  GraphGroup::load(scatterFn);
}

void AsyncGraphGroup::save(bool isFinal /* = false */) {
  // @TODO: use DefaultCommunicator as member throughout class

  auto distParams = [this]() {
    auto comm = New<DefaultCommunicator>(graphs_, /*mpi = */nullptr);
    comm->allGatherParams();
  };

  auto gatherOpt = [&](const OptimizerBase::GatherStateGetFunc& getFn) {
    io::Item data = getFn(0);
    for (size_t i = 1; i < graphs_.size(); i++)
      data.append(getFn(i));
    return data;
  };

  GraphGroup::save(isFinal, distParams, gatherOpt, /*isMainProcess=*/true);
}

void AsyncGraphGroup::finalize() {
  pool_->join_all();  // call before destructing thread pool
  pool_.reset(nullptr);
  finalized_ = true;
}

}  // namespace marian
