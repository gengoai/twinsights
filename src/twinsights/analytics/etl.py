import re
from abc import ABCMeta
from datetime import datetime
from typing import Iterable, Tuple

from twinsights.analytics.data import Post, User
from twinsights.api import Platform
from twinsights.crawler.data import CrawledDataStore, CrawledObject
from twinsights.nlp.processor import LanguageProcessor


class ETL(metaclass=ABCMeta):

    def __call__(self,
                 doc: CrawledObject,
                 db: CrawledDataStore,
                 lp: LanguageProcessor) -> Iterable[Tuple[User, Post]]:
        raise NotImplementedError()


def make_analytic_id(item_id: str):
    return f'Twitter::{item_id}'


def extract_twitter_user(doc,
                         lp: LanguageProcessor):
    return User(id=make_analytic_id(doc['id']),
                platform=Platform.Twitter,
                name=doc['name'],
                description=next(iter(lp.process([doc['description']]))),
                screen_name=doc['screen_name'],
                location=next(iter(lp.process([doc['location']]))),
                friends=doc['friends'] if 'friends' in doc else [],
                friend_count=doc['friends_count'],
                follower_count=doc['followers_count'])


class TwitterStatusETL(ETL):
    def __call__(self,
                 obj: CrawledObject,
                 db: CrawledDataStore,
                 lp: LanguageProcessor) -> Iterable[Tuple[User, Post]]:
        doc = obj.document
        is_reply = (doc['in_reply_to_status_id'] is not None
                    or doc['in_reply_to_user_id'] is not None
                    or doc['in_reply_to_screen_name'] is not None)
        source = doc['source']
        source = re.sub("^<a[^>]+>", "", source)
        source = source.replace("</a>", "").strip()
        user = extract_twitter_user(
            db.get_user(f"Twitter::user::{doc['user']['id']}").document,
            lp
        )
        post = Post(id=make_analytic_id(doc['id_str']),
                    user_id=user.id,
                    created_at=datetime.strptime(doc['created_at'],
                                                 '%a %b %d %H:%M:%S +0000 %Y'),
                    text=next(iter(lp.process([doc['full_text']]))),
                    favorite_count=doc['favorite_count'],
                    source=source,
                    is_reply=is_reply,
                    hash_tags=[a['text'] for a in doc['entities']['hashtags']])
        return [(user, post)]
