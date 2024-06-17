# GrowOVER
The source code for GrowOVER paper accepted in ACL 2024

## Dataset
You can download [GrowOVER-QA](https://drive.google.com/uc?export=download&id=1IcpH71gL1_S9BDIthhvjqj8rKE7HCp1R) and [GrowOVER-Dialogue](https://drive.google.com/file/d/1vnGBRDKeD0O9gxGC1ZUvMi7ySgbhQtoq/view?usp=sharing) here. Each zip file includes jsonl file from August to December.
![Dataset](./image.pdf)
#### Configuration
The configuration of each line in jsonl file is as follows.
```
# QA
# qa.jsonl
{ wikipedia_article_id: [
              # qa instance 1
              {
               "title": a title of Wikipedia article,
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
# article_dialogue.jsonl
{ wikipedia_article_id: [
           IDs of corresponding dialogues
          ]
}
# dialogue.jsonl
{ dialogue_id: {
          "article_id": "303",
          "created_month": 8,
          "last_modified_month": 9,
          "dialogue_type": "SAME",
          "turn": {turn number: turn_id}
          }
}
# turn.jsonl
{ turn_id: 
          {
           "dialogue_id": dialogue ID
           "turn_number": the number of this ter in the dialogue #ID,
           "user": a question,
           "expert": an answer to the user, 
           "grounded_sentence": an evidence text of the expert turn,
           "sentence_type": type of this turn (CONTRADICT / NEW / SAME),
           "sentence_index": starting index of grounded_text in the list of article sentences,
           "created_month": 8, 9, 10, 11, 12
           "created_type": type of the dialogue #ID,
          }
}           
# 
```
