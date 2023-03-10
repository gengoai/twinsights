import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import sqlalchemy

from twinsights.nlp.emoji import EmojiEntry


@dataclass
class Token:
    text: str
    upos: str
    pos: str
    lemma: str
    type: str
    idx: int
    is_stopword: bool
    props: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.idx)

    def __eq__(self,
               other):
        return self.text == other.text and self.idx == other.idx


@dataclass
class Span:
    text: str
    type: str
    label: Optional[str]
    start: int
    end: int
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document(Iterable[Token]):
    content: str
    tokens: List[Token]
    spans: List[Span]
    props: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "tokens":  [t.__dict__ for t in self.tokens],
            "spans":   [t.__dict__ for t in self.spans],
            "props":   self.props
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'Document':
        content = d['content']
        tokens = [Token(**entry) for entry in d['tokens']]
        for token in tokens:
            if "emoji" in token.props:
                token.props["emoji"] = EmojiEntry(*token.props["emoji"])
        spans = [Span(**entry) for entry in d['spans']]
        props = d['props']
        return Document(content, tokens, spans, props)

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __getitem__(self,
                    item):
        if isinstance(item, str):
            return [span for span in self.spans if
                    span.type.casefold() == item.casefold()]
        if isinstance(item, Span):
            return self.tokens[item.start:item.end]
        return self.tokens[item]

    @property
    def chunks(self) -> List[Span]:
        return self["CHUNK"]

    @property
    def entities(self) -> List[Span]:
        return self["ENTITY"]

    @property
    def sentences(self) -> List[Span]:
        return self["SENTENCE"]

    def get_overlapping_spans(self,
                              token_start: int,
                              token_end: int,
                              type: str) -> List[Span]:
        return [span for span in self.spans if
                span.start < token_end and
                span.end > token_start and span.type == type.upper()]

    def __str__(self):
        return self.content

    def __repr__(self):
        return self.content


class JsonEncodedDocument(sqlalchemy.types.TypeDecorator):
    impl = sqlalchemy.Text

    @property
    def python_type(self):
        return Document

    def process_literal_param(self,
                              value,
                              dialect):
        return self.process_bind_param(value, dialect)

    def process_bind_param(self,
                           value,
                           dialect):
        if value is None:
            return value
        return json.dumps(value.as_dict())

    def process_result_value(self,
                             value,
                             dialect):
        if value is None:
            return value
        return Document.from_dict(json.loads(value))
