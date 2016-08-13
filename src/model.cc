/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "model.h"

#include <assert.h>

#include <algorithm>

#include "args.h"
#include "utils.h"

extern Args args;

real Model::lr_ = MIN_LR;

Model::Model(Matrix& wi, Matrix& wo, int32_t hsz, real lr, int32_t seed)
            : wi_(wi), wo_(wo), hidden_(hsz), output_(wo.m_),
              grad_(hsz), rng(seed) {
  isz_ = wi.m_;
  osz_ = wo.m_;
  hsz_ = hsz;
  lr_ = lr;
  negpos = 0;
}

void Model::setLearningRate(real lr) {
  lr_ = (lr < MIN_LR) ? MIN_LR : lr;
}

real Model::getLearningRate() {
  return lr_;
}

real Model::softmax(int32_t target) {
  grad_.zero();
  output_.mul(wo_, hidden_);
  real max = 0.0, z = 0.0;
  for (int32_t i = 0; i < osz_; i++) {
    max = std::max(output_[i], max);
  }
  for (int32_t i = 0; i < osz_; i++) {
    output_[i] = exp(output_[i] - max);
    z += output_[i];
  }
  for (int32_t i = 0; i < osz_; i++) {
    real label = (i == target) ? 1.0 : 0.0;
    output_[i] /= z;
    real alpha = lr_ * (label - output_[i]);
    grad_.addRow(wo_, i, alpha);
    wo_.addRow(hidden_, i, alpha);
  }
  return -utils::log(output_[target]);
}

int32_t Model::predict(const std::vector<int32_t>& input) {
  hidden_.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden_.addRow(wi_, *it);
  }
  hidden_.mul(1.0 / input.size());

  output_.mul(wo_, hidden_);
  return output_.argmax();
}

real Model::update(const std::vector<int32_t>& input, int32_t target) {
  assert(target >= 0);
  assert(target < osz_);
  if (input.size() == 0) return 0.0;
  hidden_.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden_.addRow(wi_, *it);
  }
  hidden_.mul(1.0 / input.size());

  real loss = softmax(target);
  grad_.mul(1.0 / input.size());

  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_.addRow(grad_, *it, 1.0);
  }
  return loss;
}
