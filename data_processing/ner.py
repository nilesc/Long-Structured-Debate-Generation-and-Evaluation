import spacy

from spacy.tokens import Doc

spacy.prefer_gpu()

nlp = spacy.load('en')

verbose = False

labels_to_replace = {
    'PERSON',
    'NORP',
    'FAC',
    'ORG',
    'GPE',
    'LOC',
    'PRODUCT',
    'WORK_OF_ART',
    'PRODUCT',
    'EVENT',
    'LAW',
    'LANGAUGE',
    'DATE',
    'TIME',
    'PERCENT',
    'MONEY',
    'ORDINAL'
    'CARDINAL'
}


def replace_entities(text, replace_with='<UNK>', verbose=False):
    """
    Replace all named entities within the text.
    :param replace_with: the label with which to replace all named entity occurances. If NONE, replaces them with their
        NER label
    :return:
    """
    doc = nlp(text)

    cleaned_words = []
    previndex = 0

    for ent in doc.ents:
        label = ent.label_
        if label in labels_to_replace:
            start = ent.start
            end = ent.end
            text_to_replace = ent.text

            # Set the correct token to replace these labels with
            if replace_with is None:
                sub_label = '<{}>'.format(label)
            else:
                sub_label = replace_with

            cleaned_words += [str(token) for token in doc[previndex:start]] + [sub_label]
            previndex = start + (end - start)

            if verbose:
                print('REPLACING', text_to_replace, 'WITH', label, ':', cleaned_words)

    cleaned_words += [str(token) for token in doc[previndex:len(doc) + 2]]

    new_doc = Doc(doc.vocab, words=cleaned_words)
    return new_doc.text

