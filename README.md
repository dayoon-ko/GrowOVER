# GrowOVER
The source code for GrowOVER paper accepted in ACL 2024

## Dataset
You can download [GrowOVER-QA](https://drive.google.com/uc?export=download&id=1IcpH71gL1_S9BDIthhvjqj8rKE7HCp1R) and [GrowOVER-Dialogue](https://drive.google.com/file/d/1vnGBRDKeD0O9gxGC1ZUvMi7ySgbhQtoq/view?usp=sharing) here. Each zip file includes jsonl file from August to December.

#### Configuration
The configuration of each item in jsonl file is as follows.
```
# QA
{ wikipedia_article_id : [
                            # qa instance 1
                            {"title": a title of Wikipedia article,
                             "type": CHANGED / NEW / SAME,
                             "generated_month": "08" / "09" / "10" / "11" / "12",
                             "question": a question, 
                             "answer": an answer for the question,
                             "grounded_text": an evidence text for the answer,
                             "start_idx": starting index of grounded_text in the list of article sentences,
                             "end_idx": ending index of grounded_text in the list of article sentences,
                            },
                            # qa instance 2
                            {"title":,
                             "type":,
                             ""
                            },
                          ]
}

# Dialogue

```
