from datetime import datetime

from langdetect import detect, LangDetectException

from twinsights.crawler.data import CrawledDataStore, CrawlTask


def detect_language(text: str,
                    default_language='und') -> str:
    try:
        return detect(text).strip()
    except LangDetectException:
        return default_language


class Crawler:

    def __init__(self,
                 task: 'CrawlTask'):
        self.task = task
        self.last_run = None

    def in_timeout(self) -> bool:
        if self.last_run is None:
            return False
        current = datetime.now()
        return (current - self.last_run).seconds < self.task.timeout

    def crawl(self,
              db: CrawledDataStore) -> bool:
        raise NotImplementedError()
