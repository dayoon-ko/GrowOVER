import os
import openai


def _run_prompt(messages, temp=0):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        # model = 'gpt-3.5-turbo-1106',
        messages=messages,
        temperature=temp,
        max_tokens=256,
    )
    return messages, response


def t0_prompt(title, paragraph):
    initial_content = (
        (
            f"You will be given a paragraph about '{title}'. "
            if len(title) >= 3
            else "You will be given a paragraph. "
        )
        + "You will generate a Q&A pair to assess comprehension of its context. You will also specify a range of sentences as a 'bounding box', indicating the specific segment that contains the essential information required to FULLY INFER the answer."
    )

    prompt = [{"role": "system", "content": initial_content}]

    content = (
        f"Generate a Q&A pair based on a given context, where the context is understood but NOT DIRECTLY VISIBLE to the person answering the question. Assume the person answering the question has common sense and is aware of the details and key points in the paragraph, but the paragraph itself is not quoted or referenced directly.\n\n\
Paragraph (a list of sentences): {paragraph}\n\n"
        "Use the following instructions for generating a Q&A pair: \n 1) Provide a question, an answer, and a bounding box. \n 2) DON’T use phrases such as ‘according to the paragraph’ in your question. \n 3) An answer should be an entity or entities. \n\
 4) The bounding box for a paragraph is defined as (starting sentence index, ending sentence index): the bounding box should be sufficiently large to encompass all the information necessary for a reader to FULLY infer the answer to the question.\n 5) The sentence index starts from 0. \n\
 6) Generate a SINGLE Q&A pair.\n\n"
        "Be sure to follow the following format and write your answer within curly brackets.\n"
        "The format is as follows:\n{Question}{Answer}{starting sentence index}{ending sentence index}"
    )
    prompt.extend([{"role": "user", "content": content}])
    
    _, response = _run_prompt(prompt)
    res = response["choices"][0]["message"]["content"]

    return res


def tn_prompt(title, sentence):
    initial_content = (
        f"You will be given sentences about '{title}'. "
        if len(title) >= 3
        else "You will be given sentences. "
    ) + "You will generate a Q&A pair to assess comprehension of its context."

    prompt = [{"role": "system", "content": initial_content}]

    content = (
        f"Generate a Q&A pair based on a given context, where the context is understood but NOT DIRECTLY VISIBLE to the person answering the question. Assume the person answering the question has common sencse and is aware of the details and key points in the sentence(s), but the sentence(s) itself is not quoted or referenced directly.\n\n\
Sentence(s) : {sentence}\n\n"
        "Use the following instructions for generating a Q&A pair: \n 1) Provide a question, and an answer. \n 2) DON’T use phrases such as ‘according to the sentence(s)’ in your question. \n 3) An answer should be an entity or entities. \n \
4) Generate a SINGLE Q&A pair.\n\n"
        "Be sure to follow the following format and write your answer within curly brackets.\n"
        "The format is as follows:\n{Question}{Answer}"
    )
    prompt.extend([{"role": "user", "content": content}])

    _, response = _run_prompt(prompt)
    res = response["choices"][0]["message"]["content"]

    return res


def tn_with_surrounding_prompt(title, sentence, indicator):
    initial_content = (
        (
            f"You will be given Text list about '{title}'. "
            if len(title) >= 3
            else "You will be given sentences. "
        )
        + "You will generate a Q&A pair to assess comprehension of the Focus Sentence(s)."
    )

    prompt = [{"role": "system", "content": initial_content}]

    content = f"Text list: {sentence}\n\n" + "Focus Sentences:\n"
    for i in indicator:
        content += f"   Sentence #{i} (from Text list): {sentence[i]}\n"
    content += "\nGenerate a Q&A based on Focus Sentence(s), where the context is understood but NOT DIRECTLY VISIBLE to the person answering the question. \
You may refer to the entire Text List for broader context, but ensure the Q&A pairs are directly relevant to the Focus Sentence(s).\n\n\
Use the following instructions for generating a Q&A pair: \n 1) Provide a question, and an answer. \n 2) DON’T use phrases such as ‘according to the sentence(s)’ in your question. \n 3) An answer should be an entity or entities. \n \
4) Generate a SINGLE Q&A pair.\n\nBe sure to follow the following format and write your answer within curly brackets.\nThe format is as follows:\n{Question}{Answer}"
    prompt.extend([{"role": "user", "content": content}])

    _, response = _run_prompt(prompt)
    res = response["choices"][0]["message"]["content"]

    return res


if __name__ == "__main__":
    t0_prompt("sample title", "sample text")
    print()
    tn_prompt("sample title", "sample text")
    print()
    tn_with_surrounding_prompt("sample title", ['She considered Objectivism a systematic philosophy and laid out positions on metaphysics, epistemology, ethics, political philosophy, and aesthetics.', 'In metaphysics, Rand supported philosophical realism, and opposed anything she regarded as mysticism or supernaturalism, including all forms of religion.', 'She believed in free will as a form of agent causation and rejected determinism.', 'In epistemology, she considered all knowledge to be based on sense perception, the validity of which she considered axiomatic, and reason, which she described as "the faculty that identifies and integrates the material provided by man\'s senses".', 'She rejected all claims of non-perceptual or "a priori" knowledge, including instinct,\' \'intuition,\' \'revelation,\' or any form of \'just knowing.'], [2])