import logging
import os
import time
from operator import and_

import tweepy
from twinsights.api import Api
from twinsights.crawler import Crawler, detect_language
from twinsights.crawler.data import CrawledDataStore, CrawledObject, \
    CrawlTask, \
    Query


def create_tweepy_api_from_env() -> tweepy.API:
    auth = tweepy.OAuthHandler(os.environ['TWITTER_CONSUMER_KEY'],
                               os.environ['TWITTER_CONSUMER_SECRET'])
    auth.set_access_token(os.environ['TWITTER_ACCESS_TOKEN'],
                          os.environ['TWITTER_ACCESS_SECRET'])
    return tweepy.API(auth, wait_on_rate_limit=True)


def make_crawl_id(item_id: str,
                  is_status: bool):
    if is_status:
        return f'Twitter::status::{item_id}'
    return f'Twitter::user::{item_id}'


class TwitterUserCrawler:

    def __init__(self,
                 language: str):
        self.api = create_tweepy_api_from_env()
        self.language = language

    def crawl(self,
              db: CrawledDataStore):
        logger = logging.getLogger(self.__class__.__name__)
        for user in db.session.query(CrawledObject).filter(
                and_(CrawledObject.api == Api.TwitterUser,
                     CrawledObject.expanded.is_(False))).all():
            user_id = user.id[len('Twitter::user::'):]
            if user.expanded:
                continue
            count = 0
            try:
                for page in tweepy.Cursor(self.api.user_timeline,
                                          user_id=user_id,
                                          count=200,
                                          tweet_mode='extended').pages():
                    for status in page:
                        status = status._json
                        lang = detect_language(status['full_text'])
                        if lang == 'und':
                            lang = status['lang']
                        crawled_status = CrawledObject(
                            id=make_crawl_id(status['id'], is_status=True),
                            api=Api.TwitterStatus,
                            document=status,
                            user_id=user_id,
                            detected_lang=lang)

                        if lang == self.language:
                            db.add(crawled_status)
                            count += 1

                    logger.info("Crawled %s tweets for user %s", count, user_id)
                    if count >= 1000:
                        break
                    time.sleep(60)
                user.expanded = True
                friends = list()
                for page in tweepy.Cursor(self.api.friends,
                                          user_id=make_crawl_id(user_id, False),
                                          count=200).pages():
                    for friend in page:
                        friends.append(friend.id)
                    logger.info("Crawled %s friends for user %s", len(friends),
                                user_id)
                    time.sleep(60)
                user.document['friends'] = friends
                db.add(user)
                db.commit()

            except Exception as e:
                if 'status code = 401' in str(e) or 'Not authorized' in str(e):
                    logger.warning(e)
                elif 'status code = 404' in str(e):
                    logger.warning(e)
                elif 'TimeoutError' in str(e):
                    logger.warning("Timeout Error: Sleeping")
                    time.sleep(60 * 4)
                else:
                    logger.exception(e)
                    db.close()
                    exit()


class TweepyCrawler(Crawler):

    def __init__(self,
                 task: CrawlTask):
        super(TweepyCrawler, self).__init__(task)
        self.api = create_tweepy_api_from_env()
        self.historical = task.options.get('historical', False)
        self.ignore_retweets = task.options.get('no-retweets', False)

    def _has_more(self,
                  q: Query):
        if self.historical:
            return q.max_id is None or q.max_id != 'finished'
        return True

    def __next_results(self,
                       q: Query):
        tquery = q.text
        if self.ignore_retweets is True:
            tquery += " -filter:retweets"

        search_options = {
            'q':                tquery,
            'count':            500,
            'lang':             self.task.language,
            'tweet_mode':       'extended',
            'include_entities': True,
            'result_type':      "recent"
        }

        if self.historical:
            if q.max_id is not None and q.max_id != 'undefined':
                search_options['max_id'] = int(q.max_id) - 1
        else:
            if q.since_id is not None and q.since_id != 'undefined':
                search_options['since_id'] = int(q.since_id) + 1

        return self.api.search_tweets(**search_options)

    def __process_status(self,
                         status,
                         db: CrawledDataStore):
        lang = detect_language(status['full_text'])
        if lang == 'und':
            lang = status['lang']

        # If a task language is defined and the language of this status
        # is not that of the task language or undefined, skip the status
        if (self.task.language is not None
                and lang != self.task.language
                and lang != 'und'):
            return

        crawled_status = CrawledObject(
            id=make_crawl_id(status['id'], is_status=True),
            api=Api.TwitterStatus,
            document=status,
            detected_lang=lang)
        db.add(crawled_status)
        crawled_user = CrawledObject(
            id=make_crawl_id(status['user']['id'], is_status=False),
            api=Api.TwitterUser,
            document=status['user'],
            detected_lang=detect_language(status['user']['description']))
        db.add(crawled_user)
        db.commit()

    def __update_next_and_since(self,
                                status,
                                next_max_id,
                                next_since_id):
        if (self.historical is True
                and (next_max_id is None
                     or int(status['id']) < next_max_id)):
            next_max_id = int(status['id'])
        if (self.historical is False
                and (next_since_id is None
                     or int(status['id']) > next_since_id)):
            next_since_id = int(status['id'])

        if next_max_id is None:
            next_max_id = 'finished'
        if next_since_id is None:
            next_since_id = 'finished'

        return next_max_id, next_since_id

    def crawl(self,
              db: CrawledDataStore) -> bool:
        num_processed = 0
        logger = logging.getLogger(self.__class__.__name__)
        q: Query
        for q in filter(lambda x: self._has_more(x), self.task.queries):
            results = self.__next_results(q)
            if len(results) > 0:
                logger.info(f"Crawled %s tweets for %s", len(results), q.text)
                next_max_id = None
                next_since_id = None
                for status in (x._json for x in results):
                    self.__process_status(status, db)
                    next_max_id, next_since_id = self.__update_next_and_since(
                        status,
                        next_max_id,
                        next_since_id)

                if self.historical:
                    q.max_id = next_max_id
                elif next_since_id != 'finished':
                    q.since_id = next_since_id

                db.update_query_metadata(q.to_metadata())
                db.commit()
                num_processed += 1
                time.sleep(self.task.timeout)
            else:
                q.max_id = 'finished'

        return num_processed > 0
