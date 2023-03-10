from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Union

import empath
import sqlalchemy.orm
from sqlalchemy import Boolean, ForeignKey, Integer, Text, text, \
    TIMESTAMP
from sqlalchemy import Column, DateTime, JSON, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.sql import func

from twinsights.api import Platform
from twinsights.data import DataStore, json_encoded_dict_type
from twinsights.nlp import Document, JsonEncodedDocument
from twinsights.nlp.normalizer import Normalizer

AnalyticBase = declarative_base()

SPAN_TYPE_GETTER = {
    "token":  lambda x: x.tokens,
    "entity": lambda x: x.entities,
    "chunk":  lambda x: x.chunks
}


class User(AnalyticBase):
    __tablename__ = 'users'

    id = Column(String(1024), primary_key=True, nullable=False)
    last_updated = Column(TIMESTAMP,
                          default=sqlalchemy.func.current_timestamp())
    platform = Column(sqlalchemy.Enum(Platform), nullable=False, index=True)
    name = Column(String(1024))
    screen_name = Column(String(1024))
    description: Document = Column(JsonEncodedDocument)
    location: Document = Column(JsonEncodedDocument)
    user_metadata: dict = Column(json_encoded_dict_type)
    friends: List[str] = Column(JSON)
    friend_count: int = Column(Integer)
    follower_count: int = Column(Integer)

    def update(self,
               other: 'User'):
        merged_data = dict()
        if self.user_metadata is not None:
            merged_data.update(self.user_metadata)
        if other.user_metadata is not None:
            merged_data.update(other.user_metadata)
        if self.friends is None:
            self.friends = other.friends
            flag_modified(self, 'friends')
        if self.friend_count is None or self.friend_count != other.friend_count:
            self.friend_count = other.friend_count
        if self.follower_count is None or self.follower_count != \
                other.follower_count:
            self.follower_count = other.follower_count

        self.last_updated = func.current_timestamp()
        self.user_metadata = merged_data

    def __repr__(self):
        return f"<User(id={self.id}, name={self.name})>"


class Post(AnalyticBase):
    __tablename__ = 'posts'

    id = Column(String(1024), primary_key=True, nullable=False)
    user_id = Column(String(1024), ForeignKey('users.id'))
    created_at = Column(DateTime, nullable=False, index=True)
    text = Column(JsonEncodedDocument)
    hash_tags = Column(JSON)
    source = Column(Text)
    is_reply = Column(Boolean)
    favorite_count = Column(Integer)

    user = relationship("User", back_populates="posts")

    def update(self,
               other: 'Post'):
        self.hash_tags = list(set(self.hash_tags).union(set(other.hash_tags)))
        if other.text is not None:
            self.text = other.text
        self.favorite_count = max(self.favorite_count, other.favorite_count)

    def __repr__(self):
        return f"<Post(id={self.id})>"


User.posts = relationship("Post", order_by=Post.created_at,
                          back_populates="user")


class AnalyticDataStore(DataStore):

    def __init__(self,
                 db_file: Union[Path, str]) -> None:
        super().__init__(db_file, AnalyticBase)

    def exists(self,
               obj: Union[User, Post]) -> bool:
        if isinstance(obj, User):
            return self.session.query(User).filter(
                User.id == obj.id).first() is not None
        return self.session.query(Post).filter(
            Post.id == obj.id).first() is not None

    def hashtag_frequencies(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for t in self.session.query(Post.id, text('json_each.value')) \
                .select_from(Post, func.json_each(Post.hash_tags)) \
                .distinct(Post.id, text('json_each.value')):
            counts[t[1].lower()] += 1
        return counts

    def add_or_update(self,
                      obj: Union[User, Post]):
        if isinstance(obj, User):
            user_o: User = self.session.query(User).filter(
                User.id == obj.id).first()
            if user_o is None:
                self.session.add(obj)
            else:
                user_o.update(obj)
                self.session.merge(user_o)
        else:
            post_o: Post = self.session.query(Post).filter(
                Post.id == obj.id).first()
            if post_o is None:
                self.session.add(obj)
            else:
                post_o.update(obj)
                self.session.merge(post_o)

    def commit(self):
        self.session.commit()

    def query_posts(self,
                    query_filter):
        return self.session.query(Post).filter(query_filter).all()

    def find_post(self,
                  post_id: str):
        return self.session.query(Post).filter(Post.id == post_id).first()

    def get_user(self,
                 user_id: str) -> Union[User, None]:
        return self.session.query(User).filter(User.id == user_id).first()

    def get_post_count(self) -> int:
        return self.session.query(Post).count()

    def get_user_count(self) -> int:
        return self.session.query(User).count()

    def get_posts(self,
                  user_id: str = None,
                  hashtags: Union[Iterable[str], str] = None,
                  shuffle: bool = False,
                  limit: int = None,
                  ignore_retweets: bool = False,
                  batch_size: int = 1000):
        filters = []

        if hashtags is not None:
            if isinstance(hashtags, str):
                hashtag_list = f"'{hashtags}'"
            else:
                hashtag_list = ",".join(f"'{a}'" for a in hashtags)
            query = self.session \
                .query(Post). \
                select_from(Post, func.json_each(Post.hash_tags)) \
                .filter(text(f"json_each.value in ( {hashtag_list} )"))
        else:
            query = self.session.query(Post)

        if user_id is not None:
            query = query.filter(Post.user_id == user_id)

        if ignore_retweets:
            query = query.filter(~Post.text.like('RT @%'))

        if len(filters) > 0:
            query = query.filter(*filters)

        query = query.distinct()
        if shuffle:
            query = query.order_by(func.random())

        if limit is not None and limit > 0:
            query = query.limit(limit)

        return query.yield_per(batch_size)

    def get_users(self,
                  user_ids=None,
                  min_posts=None,
                  batch_size=1000) -> Iterable[User]:

        query = self.session.query(User)

        if min_posts is not None:
            sub_query = self.session.query(Post.user_id, func.count()).group_by(
                Post.user_id).having(
                func.count() >= min_posts).subquery()
            query = query.join(sub_query, sub_query.c.user_id == User.id)

        if user_ids is not None:
            if isinstance(user_ids, str):
                user_ids = [user_ids]
            query = query.filter(User.id.in_(user_ids))

        return query.yield_per(batch_size)

    def get_hashtags(self,
                     data_type: str) -> Iterable[Iterable[str]]:
        data_type = data_type.lower()

        if data_type == 'post':
            for post in self.get_posts():
                yield {h.lower() for h in post.hash_tags}
            return

        if data_type == 'user':
            for user in self.get_users():
                yield set(
                    [h.text.lower()[1:] for h in user.description.tokens if
                     h.type == 'HASHTAG'])
            return

        if data_type == 'user_post':
            for user in self.get_users():
                tokens = []
                for post in user.posts:
                    tokens.extend([h.lower() for h in post.hash_tags])
                yield set(tokens)
            return

        raise ValueError(f"{data_type} is invalid")

    def get_spans(self,
                  span_type: str,
                  data_type: str,
                  normalizer: Normalizer) -> Iterable[Iterable[str]]:
        data_type = data_type.lower()
        span_type = span_type.lower()

        if data_type == 'post':
            for post in self.get_posts():
                yield normalizer(post.text,
                                 SPAN_TYPE_GETTER[span_type](post.text))
            return

        if data_type == 'user':
            for user in self.get_users():
                yield normalizer(user.description,
                                 SPAN_TYPE_GETTER[span_type](user.description))
            return

        if data_type == 'user_post':
            for user in self.get_users():
                tokens = []
                for t in (normalizer(p.text,
                                     SPAN_TYPE_GETTER[span_type](p.text)) for p
                          in user.posts):
                    tokens.extend(t)
                yield tokens
            return

        raise ValueError(f"{data_type} is invalid")

    def get_empath(self,
                   data_type: str) -> Iterable[Iterable[str]]:
        data_type = data_type.lower()
        lexicon = empath.Empath()

        if data_type == 'post':
            for post in self.get_posts():
                yield lexicon.analyze(post.text.content,
                                      normalize=True)
            return

        if data_type == 'user':
            for user in self.get_users():
                yield lexicon.analyze(user.description.content,
                                      normalize=True)
            return

        if data_type == 'user_post':
            for user in self.get_users():
                categories = []
                for post in user.posts:
                    categories.extend(lexicon.analyze(post.text.content,
                                                      normalize=True))
                yield set(categories)
            return

        raise ValueError(f"{data_type} is invalid")

    def close(self):
        self.session.close()
