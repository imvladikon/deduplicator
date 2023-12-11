### Deduplication and consolidation package

Simple package to deduplicate records based on the specified attributes and to consolidate

### Deduplication usage

Usage from python:

```python
from deduplicator import Deduplicator
from deduplicator.matching import NameSimilarity

records = [{"name": ..., "phone": ...}, ...]

deduplicator = Deduplicator(
    comparators=[("name", NameSimilarity)],  # list of tuples (attribute, comparator)
    aggregation_strategy="mean",
    # attributes to block on, nested attributes can be specified with a dot
    blocking_attributes=["phone"],
    clust_kwargs={"eps": 0.1, "min_samples": 2, "metric": "precomputed"},
)
```

Optionally, you can specify a custom `blocking_rule` instead of `blocking_attributes`:

```python
from deduplicator import Deduplicator
from deduplicator.blockings import SortedNeighbourhoodBlockSplitter
from deduplicator.blockings.rules import PhoneticGroupBy, ExactGroupBy, NLetterAbbreviationGroupBy, FirstNCharsGroupBy
from deduplicator.matching import NameSimilarity

records = [{"contact_name": ..., "phone": ..., "user_id": ...}, ...]
rule1 = PhoneticGroupBy("contact_name") & ExactGroupBy("phone")
rule2 = FirstNCharsGroupBy("contact_name") & ExactGroupBy("phone")
rule3 = NLetterAbbreviationGroupBy("contact_name", n_letters=2)
blocking_rule = rule1 | rule2 | rule3
deduplicator = Deduplicator(
    comparators=[("contact_name", NameSimilarity())],
    aggregation_strategy="mean",
    blocking_rule=blocking_rule,
    blocking_splitter=SortedNeighbourhoodBlockSplitter(fields=['phone', 'user_id'],
                                                       max_block_size=128),
    clust_kwargs={"eps": 0.1, "min_samples": 2, "metric": "precomputed"},
)

for cluster_id, duplicates in deduplicator(records, similarity_threshold=0.7):
    print("-" * 100)
    print(cluster_id)

    for duplicate in duplicates:
        print(duplicate["contact_name"])

```
