# Split policy

The 39260 directed pair rows are relations, not independent IID samples.

Do not random-split pair rows.

A video can appear as a query in one edge and a demo in another; reverse edges can also exist. Random pair splitting therefore creates direct train/validation leakage.

The intended final protocol is:

1. finish near-duplicate / copy-risk clustering;
2. assign video or near-duplicate clusters to train / val / test;
3. construct edges only after the video split is fixed.

A recommended evaluation policy is:

```text
train:    query=train, demo=train
val:      query=val,   demo=train-only bank
test:     query=test,  demo=train-only bank
```

This evaluates whether an unseen query can benefit from demonstrations drawn from the training memory bank.

Before copy-risk and split-safe construction are complete, the current pair pool may be used for pipeline smoke tests or controlled overfitting checks, not for final generalization claims.
