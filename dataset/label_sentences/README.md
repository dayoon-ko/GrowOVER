```
# label same
python label_not_same.py --month_old {month_old} --root {articles_dir} --save_root {dir1}
# lable new & contradiction using fintuned roberta
python label_new_and_changed.py --month_old {month_old} --root {dir1} --save_root {dir2}
# filter contradiction with gpt4 prompting
python filter_changed.py --month_old {month_old} --root {dir2} --save_root {final_results_dir}
```