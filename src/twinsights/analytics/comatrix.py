import html
import math
import re
import unicodedata
from collections import defaultdict
from functools import partial
from typing import Any, DefaultDict, Dict, Iterable, Optional, Union

import empath
import numpy as np
import pandas as pd
import tqdm

from twinsights.analytics.data import AnalyticDataStore, Post, \
    User
from twinsights.nlp.normalizer import Normalizer


class Stats:

    def __init__(self):
        self.x1_count: Union[
            Dict[str, float], DefaultDict[str, float]] = defaultdict(float)
        self.x2_count: Union[
            Dict[str, float], DefaultDict[str, float]] = defaultdict(float)
        self.x1_sum: Union[
            Dict[str, float], DefaultDict[str, float]] = defaultdict(float)
        self.x2_sum: Union[
            Dict[str, float], DefaultDict[str, float]] = defaultdict(float)
        self.x1_x2_count: Union[
            Dict[str, Any], DefaultDict[str, Any]] = defaultdict(
            lambda: defaultdict(float))
        self.total_docs: int = 0

    def increment(self,
                  x1_iterable: Iterable[str],
                  x1_increment_by: int,
                  x2_iterable: Iterable[str],
                  x2_increment_by: int,
                  ignore_diagonal=False):
        self.total_docs += 1
        for x1 in x1_iterable:
            self.x1_count[x1] += x1_increment_by
        for x2 in x2_iterable:
            self.x2_count[x2] += x2_increment_by
            for x1 in x1_iterable:
                if ignore_diagonal and x1 == x2:
                    continue
                self.x1_x2_count[x1][x2] += 1

    def score(self,
              min_x1_count: int,
              min_x2_count: int,
              method: str):

        method = method.lower()
        final_matrix = []

        print('Pruning')
        for x1, x1_count in tqdm.tqdm(self.x1_count.items()):
            if x1_count >= min_x1_count:
                entry = dict()
                for x2, value in self.x1_x2_count[x1].items():
                    if self.x2_count[x2] >= min_x2_count:
                        entry[x2] = value
                        self.x2_sum[x2] += value
                self.x1_sum = sum(entry.values())
                self.x1_x2_count[x1] = entry
            elif x1 in self.x1_x2_count:
                del self.x1_x2_count[x1]

        print('Scoring')
        sum_scores = sum(self.x1_count.values()) + sum(self.x2_count.values())
        for x1, x1_value in tqdm.tqdm(self.x1_x2_count.items()):
            n1p = sum(x1_value.values())
            for x2, n11 in x1_value.items():
                np1 = self.x2_count[x2]
                v = 0
                idf = math.log(self.total_docs / self.x2_count[x2])
                if method == 'tf':
                    v = n11
                elif method == 'tfidf':
                    v = n11 * idf
                elif method == 'p(x2|x1)':
                    v = n11 / self.x1_count[x1]
                elif method == 'p(x1|x2)':
                    v = n11 / self.x2_count[x2]
                elif method == 'p':
                    v = n11 / sum(x1_value.values())
                elif method == 'ppmi':
                    v = max(math.log2(n11) - math.log2(
                        (n1p * np1) / sum_scores
                    ), 0)
                elif method == 'tscore':
                    v = (n11 - (n1p * np1) / sum_scores) / math.sqrt(n11)
                elif method == 'npmi':
                    v = math.log2(n11) - math.log2(
                        (n1p * np1) / sum_scores
                    )
                    v /= -math.log2(n11 / sum_scores)
                else:
                    raise KeyError('Invalid scoring method %s', method)
                final_matrix.append({"item1": x1, "item2": x2, "score": v})

        return pd.DataFrame(final_matrix).fillna(0)


class CoMatrix:

    def __init__(self,
                 name: str,
                 db: AnalyticDataStore,
                 normalizer: Optional[Normalizer]):
        self.name = name
        self.db = db
        self.normalizer = normalizer
        self.matrix: Optional[pd.DataFrame] = None
        self.ITEM_TYPES = {
            "hashtag":  CoMatrix.__get_hashtags,
            "token":    partial(self.__get_spans, get_func=lambda x: x.tokens),
            "chunk":    partial(self.__get_spans, get_func=lambda x: x.chunks),
            "entity":   partial(self.__get_spans,
                                get_func=lambda x: x.entities),
            "mention":  CoMatrix.__get_mentions,
            "username": CoMatrix.__get_username,
            "empath":   CoMatrix.__get_empath
        }

    def __get_spans(self,
                    x: Union[User, Post],
                    count_by: str,
                    get_func) -> Iterable[str]:
        items = []
        if count_by == 'post':
            items = self.normalizer(x.text, get_func(x.text))
        elif count_by == 'user_post':
            for post in x.posts:
                items.extend(self.normalizer(post.text, get_func(post.text)))
        elif count_by == 'user':
            items = self.normalizer(x.description, get_func(x.description))
        return items

    @staticmethod
    def _get_text(x: Union[User, Post],
                  data_type: str) -> str:
        if data_type == 'post':
            return x.text.content
        elif data_type == 'user':
            return x.description.content
        elif data_type == 'user_post':
            return ' '.join((post.text.content for post in x.posts))
        return ""

    @staticmethod
    def __get_empath(x: Union[User, Post],
                     data_type: str) -> Iterable[str]:
        lexicon = empath.Empath()
        categories = []
        for k, v in lexicon.analyze(CoMatrix._get_text(x, data_type),
                                    normalize=False).items():
            for i in range(int(v)):
                categories.append(k)
        return categories

    @staticmethod
    def __get_hashtags(x: Union[User, Post],
                       data_type: str) -> Iterable[str]:
        if data_type == 'post':
            return set(
                [unicodedata.normalize('NFKC', html.unescape(t).lower()) for t
                 in
                 x.hash_tags])
        elif data_type == 'user_post':
            tags = []
            for post in x.posts:
                tags.extend(
                    [unicodedata.normalize('NFKC', html.unescape(t).lower()) for
                     t
                     in post.hash_tags])
            return tags
        elif data_type == 'user':
            return set(
                t.text.lower() for t in x.description if t.type == 'HASHTAG')
        return []

    @staticmethod
    def __get_mentions(x: Union[User, Post],
                       data_type: str) -> Iterable[str]:
        if data_type == 'post':
            return set(re.findall(r'@\S+', x.text.content))
        elif data_type == 'user_post':
            mentions = []
            for post in x.posts:
                mentions.extend(set(re.findall(r'@\S+', post.text.content)))
            return mentions
        elif data_type == 'user':
            return set(re.findall(r'@\S+', x.description.content))
        return []

    @staticmethod
    def __get_username(x: Union[User, Post],
                       data_type: str) -> Iterable[str]:
        if data_type == 'post':
            return [x.user.name]
        elif data_type == 'user_post':
            return [x.name]
        elif data_type == 'user':
            return [x.name]
        return []

    @staticmethod
    def __get_increment(item: str,
                        count_by: str,
                        datum: Union[User, Post]):
        if item == 'user_id' and count_by == 'user_post':
            return len(datum.posts)
        return 1

    def build(self,
              item1: str,
              item2: str,
              min_item1_count: int = 1,
              min_item2_count: int = 1,
              count_by: str = 'post',
              scorer: str = 'tf'):
        item1 = item1.lower()
        item2 = item2.lower()
        count_by = count_by.lower()
        scorer = scorer.lower()

        if count_by not in ("post", "user", "user_post"):
            raise ValueError(
                f"Invalid count_by parameter '{count_by}' should be one of "
                f"'post', 'user', or 'user_post'")

        data_stream = self.db.get_posts() if count_by == 'post' else \
            self.db.get_users()

        total_count = self.db.get_post_count() if count_by == 'post' else \
            self.db.get_user_count()

        stats = Stats()
        print("Collecting Statistics")
        for datum in tqdm.tqdm(data_stream, total=total_count):
            item1_data = self.ITEM_TYPES[item1](datum, count_by)
            item2_data = item1_data if item1 == item2 else \
                self.ITEM_TYPES[item2](datum, count_by)
            item1_increment = CoMatrix.__get_increment(item1, count_by, datum)
            item2_increment = CoMatrix.__get_increment(item2, count_by, datum)
            stats.increment(item1_data,
                            item1_increment,
                            item2_data,
                            item2_increment,
                            item1 == item2)

        self.matrix = stats.score(min_item1_count,
                                  min_item2_count,
                                  scorer)
        self.matrix.to_sql(self.name,
                           con=self.db.engine,
                           index=False,
                           if_exists="replace")

    @staticmethod
    def load(name: str,
             db: AnalyticDataStore) -> 'CoMatrix':
        comatrix = CoMatrix(name=name,
                            db=db,
                            normalizer=None)
        comatrix.matrix = pd.read_sql(f"SELECT * FROM {comatrix.name}",
                                      con=comatrix.db.engine)
        comatrix.matrix = pd.pivot_table(comatrix.matrix,
                                         values="score",
                                         columns="item2",
                                         index="item1",
                                         aggfunc=np.sum).fillna(0)
        return comatrix
