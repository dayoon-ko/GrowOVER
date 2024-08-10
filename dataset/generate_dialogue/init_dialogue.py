import logging
from argparse import Namespace

from app.set_init import init_settings, get_args
from app.data_loader import InitDialogueDataLoader
from app.data_saver import DataSaver
from app.dialogue_generator import DialogueGenerator
from app.paragraph_selector import ParagraphSelector


def main(args: Namespace):
    data_loader = InitDialogueDataLoader(args)
    data_saver = DataSaver(args)
    dialogue_generator = DialogueGenerator(args)
    paragraph_selector = ParagraphSelector(args)
    cnt, total_created_dialogue = 0, 0
    
    # set data_saver for unique id
    dialogue_id, turn_id = data_loader.dialogue_id, data_loader.turn_id
    data_saver.set_id(dialogue_id, turn_id)
    
    print(f"Initial Dialogue Generation Start at {args.month} Month!!")
    logging.info(f"Initial Dialogue Generation Start at {args.month} Month!!")
    
    for id, wiki_data in data_loader:
        cnt += 1
        url, title, article = wiki_data
        logging.info(f"----------------- [ {cnt} ] article {id} (title: {title}) -----------------")
        
        # get informative articles 
        # sentences(list, [sentence_0, sentence_1, ...]), start_idx(int), sentence_type(list, ["SAME", "NEW", "CONTRADICT", "SAME", ...])
        target_paragraphs = paragraph_selector.select_paragraphs_for_initialize(id, article)
        if len(target_paragraphs) == 0:
            logging.warning(f"No informative paragraphs in article")
            continue
        
        # generate dialogues
        dialogues = dialogue_generator.run(id, title, target_paragraphs)
    
        # save dialogue
        data_saver.save_created_dialogue(dialogues)
        
        total_created_dialogue += len(dialogues)
        
    logging.info(f"Total created dialogues: {total_created_dialogue}")


if __name__ == "__main__":
    init_settings()
    args = get_args()
    
    main(args)