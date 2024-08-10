import logging
from argparse import Namespace

from app.set_init import init_settings, get_args
from app.data_loader import UpdateDialogueDataLoader
from app.data_saver import DataSaver
from app.dialogue_generator import DialogueGenerator
from app.dialogue_manager import DialogueManager
from app.paragraph_selector import ParagraphSelector
from app.utils import article_id_is_not_consistent, article_id, dialogue_types

def main(args: Namespace):
    data_loader = UpdateDialogueDataLoader(args)
    data_saver = DataSaver(args)
    deleted_data_saver = DataSaver(args, deleted=True)
    dialogue_manager = DialogueManager(args)
    dialogue_generator = DialogueGenerator(args)
    paragraph_selector = ParagraphSelector(args)
    cnt, total_created_dialogue = 0, 0
    
    # set data_saver for unique id
    dialogue_id, turn_id = data_loader.dialogue_id, data_loader.turn_id
    data_saver.set_id(dialogue_id, turn_id)

    print(f"Update Dialogue Generation Start at {args.month} Month!!")
    logging.info(f"Update Dialogue Generation Start at {args.month} Month!!")
    for id, wiki_data, old_label, new_label, old_dialogue, old_turn in data_loader:
        if old_label is None or new_label is None:
            continue
        
        cnt += 1
        
        url, title, article = wiki_data
        logging.info(f"----------------- [ {cnt} ] article {id} (title: {title}) -----------------")
        
        # update dialogue
        maintained_dialogue, maintained_turn, deleted_dialogue, deleted_turn = dialogue_manager.update_dialogue(title, old_dialogue, old_turn, old_label)
        data_saver.save_updated_dialogue(maintained_dialogue, maintained_turn)
        deleted_data_saver.save_updated_dialogue(deleted_dialogue, deleted_turn)
        logging.info(f"\tSAME Dialogues: {len(maintained_dialogue)}\tDELETED Dialogues: {len(deleted_dialogue)}")

        # get informative articles
        target_paragraphs = paragraph_selector.select_paragraphs_for_update(article, new_label, old_label)
        if len(target_paragraphs) == 0:
            logging.warning(f"No informative paragraphs or there is no updated paragraphs in article")
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