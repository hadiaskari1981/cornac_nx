# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import operator
import datetime
import dateutil.relativedelta
from tqdm import tqdm
from .base_method import BaseMethod


class TimeSplit(BaseMethod):
    """Splitting data into training, validation, and test sets based on provided sizes.
    Data is always shuffled before split.

    Parameters
    ----------
    data: array-like, required
        Raw preference data in the triplet format [(user_id, item_id, rating_value)].

    test_size: float, optional, default: 0.2
        The proportion of the test set, \
        if > 1 then it is treated as the size of the test set.

    val_size: float, optional, default: 0.0
        The proportion of the validation set, \
        if > 1 then it is treated as the size of the validation set.

    rating_threshold: float, optional, default: 1.0
        Threshold used to binarize rating values into positive or negative feedback for
        model evaluation using ranking metrics (rating metrics are not affected).

    seed: int, optional, default: None
        Random seed for reproducibility.

    exclude_unknowns: bool, optional, default: True
        If `True`, unknown users and items will be ignored during model evaluation.

    verbose: bool, optional, default: False
        Output running log.

    """

    def __init__(
        self,
        data,
        rating_threshold=1.0,
        seed=None,
        exclude_unknowns=True,
        verbose=False,
        **kwargs,
    ):
        super().__init__(
            data=data,
            rating_threshold=rating_threshold,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            **kwargs,
        )

        self._split()

    def _split(self):

        # data.sort(key=operator.itemgetter(4))

        max_timestamp = max(self.data, key=operator.itemgetter(3))[3]
        max_timestamp = datetime.datetime.strptime(max_timestamp, "%Y-%m-%d")
        split_point = max_timestamp - dateutil.relativedelta.relativedelta(months=1)
        split_point = split_point.strftime("%Y-%m-%d")

        train_data = [t for t in tqdm(self.data) if t[3] < split_point]
        test_data = [t for t in tqdm(self.data) if t[3] >= split_point]
        val_data = None

        self.build(train_data=train_data, test_data=test_data, val_data=val_data)
