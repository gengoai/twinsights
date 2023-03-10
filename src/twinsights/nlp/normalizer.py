from typing import Dict, Iterable, List, Union

from twinsights.nlp import Document, Span, Token

_TOKEN_TO_STR = {
    ".text":              lambda x: x.text,
    ".text.type":         lambda x: f"{x.text}/{x.type}",
    ".text.pos":          lambda x: f"{x.text}/{x.pos}",
    ".text.upos":         lambda x: f"{x.text}/{x.upos}",
    ".lower(text)":       lambda x: x.text.lower(),
    ".lower(text.pos)":   lambda x: f"{x.text.lower()}/{x.pos}",
    ".lower(text.upos)":  lambda x: f"{x.text.lower()}/{x.upos}",
    ".lower(text.type)":  lambda x: f"{x.text.lower()}/{x.type}",
    ".type":              lambda x: f"TYPE:{x.type}",
    ".lemma":             lambda x: x.lemma,
    ".lemma.pos":         lambda x: f"{x.lemma}/{x.pos}",
    ".lemma.upos":        lambda x: f"{x.lemma}/{x.upos}",
    ".lower(lemma)":      lambda x: x.lemma.lower(),
    ".lower(lemma.pos)":  lambda x: f"{x.lemma.lower()}/{x.pos}",
    ".lower(lemma.upos)": lambda x: f"{x.lemma.lower()}/{x.upos}",
    ".emoji.name":        lambda x: f"EMOJI:{x.props['emoji'].name}",
    ".emoji.group":       lambda x: f"EMOJI:{x.props['emoji'].group}",
    ".emoji.sub_group":   lambda x: f"EMOJI:{x.props['emoji'].sub_group}",
}


class Normalizer:

    def __init__(self,
                 ignore_stopwords: bool = True,
                 default_replacement: str = ".lower(text)",
                 replacements: Dict[str, str] = None,
                 ignored_types: Iterable[str] = None):
        self.ignore_stopwords = ignore_stopwords
        self.default_replacement = default_replacement
        self.replacements = dict() if replacements is None else replacements
        self.ignored_types = set() if ignored_types is None else set(
            ignored_types)

    def __to_string(self,
                    tokens: List[Token]) -> str:
        strings = []
        for token in tokens:
            replace = self.replacements.get(token.type,
                                            self.default_replacement)
            if replace.startswith("."):
                string = _TOKEN_TO_STR[replace](token)
            else:
                string = replace
            strings.append(string)
        return " ".join(strings)

    def __call__(self,
                 document: Document,
                 spans_or_tokens: Iterable[Union[Span, Token]]) \
            -> Iterable[str]:
        return_value = []
        for item in spans_or_tokens:
            tokens = [item] if isinstance(item, Token) \
                else document[item.start:item.end]

            if (self.ignore_stopwords
                    and (any((t.is_stopword for t in tokens))
                         or len(tokens) == 0
                         or any(len(t.text.strip()) == 0 for t in tokens))):
                continue

            if any((t.type in self.ignored_types for t in tokens)):
                continue

            return_value.append(self.__to_string(tokens))

        return return_value
