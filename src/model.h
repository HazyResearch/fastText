/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_MODEL_H
#define FASTTEXT_MODEL_H

#include <vector>
#include <random>

#include "matrix.h"
#include "vector.h"
#include "real.h"

class Model {
  private:
    Matrix& wi_;
    Matrix& wo_;
    Vector hidden_;
    Vector output_;
    Vector grad_;
    int32_t hsz_;
    int32_t isz_;
    int32_t osz_;

    static real lr_;

    size_t negpos;
    std::vector< std::vector<int32_t> > paths;
    std::vector< std::vector<bool> > codes;

    static constexpr real MIN_LR = 0.000001;

  public:
    Model(Matrix&, Matrix&, int32_t, real, int32_t);

    void setLearningRate(real);
    real getLearningRate();

    real softmax(int32_t);

    int32_t predict(const std::vector<int32_t>&);
    real update(const std::vector<int32_t>&, int32_t);

    std::minstd_rand rng;
};

#endif
