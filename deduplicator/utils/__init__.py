#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from deduplicator.utils.common_utils import new_id
from deduplicator.utils.iter_utils import (batched, iter_pairwise, all_pairs_for,
                                           all_pairs_for_range, flatten)
from deduplicator.utils.reflection_utils import (check_backend_availability, is_package_available, iter_subclasses)
from deduplicator.utils.er_utils import *
from deduplicator.utils.logging_utils import *
