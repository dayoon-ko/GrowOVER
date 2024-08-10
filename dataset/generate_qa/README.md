```
# 1. Create initial QA pairs
python create_t0.py --t0_month 8 --wiki_dump_root {articles_dir} --save_root {save_dir} --n_clusters {num clusters}

# 2. Temporal updates with each new wiki dump
python create_tn.py --month_old {old month} --wiki_dump_root {articles_dir} --label_new_root {dir1 in sentence labeling} --label_chg_root {dir2 in sentence labeling} --save_root {save_dir} --n_clusters {num clusters}
```