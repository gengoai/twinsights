import html
import re
import unicodedata
from typing import Any, Dict, Iterable, List

import spacy
from spacy.matcher import Matcher
from spacy.tokens.doc import Doc

from twinsights.conf import Config
from twinsights.nlp import Document, Span, Token
from twinsights.nlp.emoji import get_emoji_dict

DIGIT_PATTERN = re.compile(r"\d+([,.]\d+)*")


class LanguageProcessor:

    def __init__(self,
                 language: str = "en",
                 patterns: Dict[str, List[List[Dict[str, Any]]]] = None):
        self.nlp: spacy.language.Language = spacy.load(
            Config["spacy.models"][language]
        )
        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add('HASHTAG', [[{'ORTH': '#'},
                                      {'IS_PUNCT': False}]])
        if patterns is not None:
            for name, pattern in patterns.items():
                self.matcher.add(name, pattern)

    @staticmethod
    def get_token_type(token: spacy.tokens.Token,
                       token_types: Dict[int, str]) -> str:
        text = token.text.lower()
        category = "OTHER" if len(text) != 1 \
            else unicodedata.category(token.text)

        if token.is_sent_start and token.text == 'RT':
            return "RETWEET"

        if token.idx in token_types:
            return token_types[token.idx]

        if (text.startswith("http")
                or text.startswith("www")
                or text.endswith(".com")):
            return "URL"

        if (token.text.startswith("@")
                and len(token.text) > 1
                and token.text[1:].isalnum()):
            return "MENTION"

        if token.is_punct:
            return "PUNCT"

        if token.is_alpha:
            return "ALPHA"

        if (token.is_digit
                or token.pos_ == 'NUM'
                or DIGIT_PATTERN.match(token.text)):
            return "DIGIT"

        if token.text in get_emoji_dict():
            return "EMOJI"

        if category.startswith('S'):
            return "SYM"
        if category.startswith('P'):
            return "PUNCT"
        if category.startswith('M'):
            return "MARK"

        return "OTHER"

    def __build_tokens(self,
                       doc: spacy.tokens.Doc) -> List[Token]:
        token_types: Dict[int, str] = dict()
        matches = self.matcher(doc)

        span_info = dict()
        spans = []
        for match_id, start, end in matches:
            span = doc[start:end]
            spans.append(span)
            span_info[span] = (match_id, start, end)
        spans = spacy.util.filter_spans(spans)

        with doc.retokenize() as retokenizer:
            for span in spans:
                match_id, start, end = span_info[span]
                token_types[doc[start].idx] = self.nlp.vocab.strings[
                    match_id]
                retokenizer.merge(span)

        tokens = []
        for token in doc:
            if token.is_space:
                continue
            props = dict()
            token_type = LanguageProcessor.get_token_type(token,
                                                          token_types)
            if token_type == 'EMOJI':
                props["emoji"] = get_emoji_dict()[token.text]

            is_stopword = token.is_stop or token_type in (
                "SYM",
                "OTHER",
                "PUNCT",
                "MARK",
                "DIGIT"
            ) or (token_type == 'ALPHA' and len(token.text) == 1)

            tokens.append(Token(text=token.text,
                                upos=token.pos_,
                                pos=token.tag_,
                                lemma=token.lemma_,
                                type=token_type,
                                is_stopword=is_stopword,
                                idx=token.idx,
                                props=props))
        return tokens

    def process(self,
                texts: Iterable[str]) -> Iterable[Document]:
        for doc in self.nlp.pipe(
                (unicodedata.normalize('NFKC', html.unescape(text)) for text in
                 texts)
        ):
            tokens = self.__build_tokens(doc)
            spans = []
            for sent in doc.sents:
                spans.append(Span(text=sent.text,
                                  type="SENTENCE",
                                  label=None,
                                  start=sent.start,
                                  end=sent.end))
            for ent in doc.ents:
                spans.append(Span(text=ent.text,
                                  type="ENTITY",
                                  label=ent.label_,
                                  start=ent.start,
                                  end=ent.end))
            for chunk in doc.noun_chunks:
                spans.append(Span(text=chunk.text,
                                  type="CHUNK",
                                  label="NP",
                                  start=chunk.start,
                                  end=chunk.end))
            yield Document(str(doc), tokens, spans)
