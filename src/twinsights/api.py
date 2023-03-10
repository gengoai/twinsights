from enum import Enum
from typing import TYPE_CHECKING

from twinsights.conf import Config
from twinsights.reflection import get_class_for_name

if TYPE_CHECKING:
    from twinsights.crawler.data import CrawlTask
    from twinsights.crawler import Crawler
    from twinsights.analytics.etl import ETL


class Platform(Enum):
    Twitter = 'Twitter'
    Reddit = 'Reddit'


class Api(Enum):
    TwitterStatus = 'TwitterStatus'
    TwitterUser = 'TwitterUser'

    def etl(self) -> 'ETL':
        return get_class_for_name(Config[f"Api.{self.name}.ETL"])()

    def crawler(self,
                task: 'CrawlTask') -> 'Crawler':
        crawler = get_class_for_name(Config[f"Api.{self.name}.Crawler"])(task)
        return crawler

    @property
    def platform(self) -> Platform:
        if self is Api.TwitterStatus or self is Api.TwitterUser:
            return Platform.Twitter
        raise ValueError()
