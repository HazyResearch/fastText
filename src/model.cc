/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "model.h"

#include <math.h>

#include "args.h"
#include "utils.h"

extern Args args;

real Model::lr_ = MIN_LR;

Model::Model(Matrix& wi, Matrix& wo, int32_t hsz, real lr, int32_t seed)
            : wi_(wi), wo_(wo), hidden_(hsz), output_(wo.m_),
              grad_(hsz) {
  isz_ = wi.m_;
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

real Model::logistic_loss(real marginal) {
  grad_.zero();
  output_.mul(wo_, hidden_);
  real h = 1.0 / (1.0 + exp(-output_[0]));
  real alpha = lr_ * (h - marginal);
  grad_.addRow(wo_, 0, alpha);
  wo_.addRow(hidden_, 0, alpha);
  return -(marginal * utils::log(output_[0]) +
           (1.0-marginal) * utils::log(1.0 - output_[0]));
}

real Model::predict(const std::vector<int32_t>& input) {
  hidden_.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden_.addRow(wi_, *it);
  }
  hidden_.mul(1.0 / input.size());

  output_.mul(wo_, hidden_);
  return output_[0];
}

real Model::update(const std::vector<int32_t>& input, real marginal) {
  if (input.size() == 0) return 0.0;
  hidden_.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden_.addRow(wi_, *it);
  }
  hidden_.mul(1.0 / input.size());

  real loss = logistic_loss(marginal);
  grad_.mul(1.0 / input.size());

  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_.addRow(grad_, *it, 1.0);
  }
  return loss;
}
