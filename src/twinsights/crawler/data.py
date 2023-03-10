from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import sqlalchemy
from sqlalchemy import Boolean, Column, String
from sqlalchemy.orm import declarative_base

from twinsights.api import Api
from twinsights.data import DataStore, json_encoded_dict_type

CrawlerBase = declarative_base()


@dataclass
class Query:
    id: str
    text: str
    since_id: str
    max_id: str

    def __repr__(self):
        return f"<Query(id='{self.id}', query='{self.text}')>"

    def to_metadata(self) -> 'QueryMetadata':
        return QueryMetadata(id=self.id, since_id=self.since_id, max_id=self.max_id)


@dataclass
class CrawlTask:
    language: str
    api: Api
    name: str
    timeout: int
    options: Dict[str, Any] = field(default_factory=dict)
    queries: List[Query] = field(default_factory=list)

    def __repr__(self):
        return f"<CrawlTask(name={self.name}, api={self.api}, queries=" \
               f"{self.queries})>"


class QueryMetadata(CrawlerBase):
    __tablename__ = 'query_metadata'

    id = Column(String(1024), primary_key=True, nullable=False)
    since_id = Column(String(1024), nullable=True)
    max_id = Column(String(1024), nullable=True)

    def __repr__(self):
        return f"<QueryMetadata(id={self.id}, since_id={self.since_id}, " \
               f"max_id={self.max_id})>"


class CrawledObject(CrawlerBase):
    __tablename__ = 'data'

    id = Column(String(1024), primary_key=True, nullable=False)
    api = Column(sqlalchemy.Enum(Api), nullable=False, index=True)
    expanded = Column(Boolean, default=False, index=True)
    document = Column(json_encoded_dict_type, nullable=False)
    user_id = Column(String(1024), index=True)
    detected_lang = Column(String(4), index=True)

    def __repr__(self):
        return f"<CrawledObject(id={self.id}, api={self.api})>"


class CrawledDataStore(DataStore):

    def __init__(self,
                 db_file: Union[Path, str]) -> None:
        super().__init__(db_file, CrawlerBase)

    def restore_query_metadata(self,
                               task: 'CrawlTask'):
        for q in task.queries:
            qm = self.session.query(QueryMetadata).filter(
                QueryMetadata.id == q.to_metadata().id).first()
            if qm is not None:
                if q.max_id is not None and q.max_id == 'undefined':
                    q.max_id = qm.max_id
                else:
                    q.max_id = None
                if q.max_id is not None and q.since_id == 'undefined':
                    q.since_id = qm.since_id
                else:
                    q.max_id = None
            else:
                if q.max_id == 'undefined':
                    q.max_id = None
                if q.since_id == 'undefined':
                    q.since_id = None
                self.session.add(q.to_metadata())
                self.session.commit()

    def commit(self):
        self.session.commit()

    def close(self):
        self.session.close()

    def get_user(self,
                 user_id: str) -> CrawledObject:
        return self.session.query(CrawledObject).filter(
            CrawledObject.id == user_id).first()

    def status_count(self):
        return self.session.query(CrawledObject).filter(
            CrawledObject.api != Api.TwitterUser).count()

    def get_data(self) -> List[CrawledObject]:
        return self.session.query(CrawledObject).filter(
            CrawledObject.api != Api.TwitterUser).yield_per(500)

    def find(self,
             id: str) -> Union[CrawledObject, None]:
        return self.session.query(CrawledObject).filter(
            CrawledObject.id == id).first()

    def update_query_metadata(self,
                              qm: QueryMetadata):
        self.session.merge(qm)

    def add(self,
            obj: CrawledObject):
        if obj.api == Api.TwitterUser:
            current = self.session.query(CrawledObject).filter(
                CrawledObject.id == obj.id).first()
            obj.expanded = False if current is None else current.expanded
        self.session.merge(obj)
