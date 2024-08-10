```
# 1. Label unchanged sentences
python label_not_same.py --month_old {month_old} --root {articles_dir} --save_root {dir1}

# 2. Label new & changed sentences (w/ roberta-mnli-finetuned)
python label_new_and_changed.py --month_old {month_old} --root {dir1} --save_root {dir2}

# 3. Verify changed sentences (w/ GPT-4)
python filter_changed.py --month_old {month_old} --root {dir2} --save_root {final_results_dir}
```