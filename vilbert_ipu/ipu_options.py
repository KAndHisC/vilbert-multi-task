# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import poptorch


def get_options(config=None)->poptorch.Options:
    logger = logging.getLogger(__name__)
    '''
    Set ipu specific options for the model, see documentation:
    https://docs.graphcore.ai/en/latest/
    '''
    if poptorch.ipuHardwareVersion() != 2:
        raise RuntimeError("This version of BERT requires an IPU Mk2 system to run.")
    ## TODO--
    custom_opts = poptorch.Options()
    custom_opts.deviceIterations(1)
    custom_opts.replicationFactor(1)
    custom_opts.Training.gradientAccumulation(7)
    custom_opts.randomSeed(99)
    return custom_opts

opts = get_options()