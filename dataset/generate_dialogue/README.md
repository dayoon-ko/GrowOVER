```
# 1. Create initial QA pairs
python init_dialogue.py --wiki_dir {articles_dir} --label-dir {dir2 in sentence labeling} --dialogue-dir {save_dir}

# 2. Temporal updates with each new wiki dump
python update_dialogue.py --wiki_dir {articles_dir} --label-dir {dir2 in sentence labeling} --dialogue-dir {save_dir} --month {month old}
```