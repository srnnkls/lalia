# Note for developers

To recreate and thus update hatch env dependencies, run:

```bash
./recreate_hatch_env.sh
```

or specify a specific env name

```bash
./recreate_hatch_env.sh development
```

After updating black to `>=24`, there have been changes in its formatting behaviour. To ignore the reformat commit in `git blame`, either call it with the provided ignore file

```bash
git blame important.py --ignore-revs-file .git-blame-ignore-revs
```

or set it as a default

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```
