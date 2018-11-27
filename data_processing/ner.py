import spacy

from spacy.tokens import Doc

# To Download spacy english dictionary run:
# python -m spacy download en

spacy.prefer_gpu()

nlp = spacy.load('en')

"""
# https://spacy.io/usage/linguistic-features

PERSON	People, including fictional.
NORP	Nationalities or religious or political groups.
FAC	Buildings, airports, highways, bridges, etc.
ORG	Companies, agencies, institutions, etc.
GPE	Countries, cities, states.
LOC	Non-GPE locations, mountain ranges, bodies of water.
PRODUCT	Objects, vehicles, foods, etc. (Not services.)
EVENT	Named hurricanes, battles, wars, sports events, etc.
WORK_OF_ART	Titles of books, songs, etc.
LAW	Named documents made into laws.
LANGUAGE	Any named language.
DATE	Absolute or relative dates or periods.
TIME	Times smaller than a day.
PERCENT	Percentage, including "%".
MONEY	Monetary values, including unit.
QUANTITY	Measurements, as of weight or distance.
ORDINAL	"first", "second", etc.
CARDINAL	Numerals that do not fall under another type.
"""


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
    # 'DATE',
    # 'TIME',
    # 'PERCENT',
    # 'MONEY',
    # 'ORDINAL'
    # 'CARDINAL'
}


def replace_entities(text, replace_with='<UNK>'):
    """
    Replace all named entities within the text.
    :param replace_with: the label with which to replace all named entity occurances. If NONE, replaces them with their
    :return:
    """
    doc = nlp(text)

    cleaned_words = [t.text for t in doc]

    for ent in doc.ents:
        label = ent.label_
        if label in labels_to_replace:
            start = ent.start
            end = ent.end
            text_to_replace = ent.text

            if replace_with is None:
                replace_with = label

            cleaned_words[start:end] = [replace_with] * (end - start)

            if verbose:
                print('REPLACING', text_to_replace, 'WITH', label, ':', cleaned_words)

    new_doc = Doc(doc.vocab, words=cleaned_words)
    return new_doc.text


if __name__ == '__main__':
    text = (u"When Sebastian Thrun started working on self-driving cars at "
            u"Google in 2007, few people outside of the company took him "
            u"seriously. “I can tell you very senior CEOs of major American "
            u"car companies would shake my hand and turn away because I wasn’t "
            u"worth talking to,” said Thrun, now the co-founder and CEO of "
            u"online higher education startup Udacity, in an interview with "
            u"Recode earlier this week.")

    new = replace_entities(text)
    print(new)
