from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from twinsights.analytics.data import AnalyticDataStore
from twinsights.api import Api
from twinsights.crawler.data import CrawledDataStore, CrawlTask, Query
from twinsights.nlp.normalizer import Normalizer
from twinsights.nlp.processor import LanguageProcessor


class Project:

    def __init__(self,
                 path: Path):
        self.path: Path = path
        self.crawl_tasks: Dict[str, CrawlTask] = {}
        self.queries: List[str] = []
        self.language: str = 'en'
        self.comatrices: Dict[str, Dict[str, Any]] = dict()
        self.cluster_parameters: Dict[str, Dict[str, Any]] = dict()
        self.tokenizer_parameters: Dict[str, Any] = dict()
        self.normalizers: Dict[str, Dict[str, Any]] = dict()
        self.hashtag_labels: Dict[str, Any] = dict()

        with open(path / 'config.yml') as fp:
            doc = yaml.load(fp, Loader=yaml.FullLoader)
            self.name = doc.get('name', '')
            self.description = doc.get('description', '')
            self.settings = doc.get('settings', dict())
            self.language = doc.get('language', 'en')

            analysis = doc.get("analysis", dict())
            self.comatrices = analysis.get('comatrix', dict())

            default_parameters = self.comatrices.get('default_parameters', {
                "min_item1_count": 1,
                "min_item2_count": 1,
                "count_by":        "post",
                "scorer":          "npmi"
            })
            if 'default_parameters' in self.comatrices:
                del self.comatrices['default_parameters']
            for comatrix in self.comatrices.values():
                for key, value in default_parameters.items():
                    if key not in comatrix:
                        comatrix[key] = value

            self.cluster_parameters = analysis.get('clusters', dict())
            for comatrix in self.comatrices.keys():
                if comatrix not in self.cluster_parameters:
                    self.cluster_parameters[comatrix] = dict()

            default_parameters = self.cluster_parameters.get(
                'default_parameters', {
                    "k":         -1,
                    "normalize": False
                })
            if 'default_parameters' in self.cluster_parameters:
                del self.cluster_parameters['default_parameters']
            for cluster in self.cluster_parameters.values():
                for key, value in default_parameters.items():
                    if key not in cluster:
                        cluster[key] = value

            nlp = doc.get('nlp', dict())
            self.token_matchers = nlp.get('token_matchers', dict())
            self.normalizers = nlp.get('normalizers', dict())
            if "default" not in self.normalizers:
                self.normalizers["default"] = dict()

            self.hashtag_labels = doc.get('labeler', dict())
            for index, task in enumerate(doc.get('crawler_tasks', [])):
                api = Api(task['api'])
                name = task.get('name', f'Crawl_{index}')
                language = task.get('language', self.language)
                timeout = task.get('timeout', self.settings.get('timeout', 60))
                options = task.get('options', dict())
                queries = task['queries']
                task = CrawlTask(api=api,
                                 name=name,
                                 timeout=timeout,
                                 options=options,
                                 language=language,
                                 queries=[Query(id=f"{name}::{q}",
                                                text=q,
                                                since_id='undefined',
                                                max_id='undefined') for q in
                                          queries])
                self.crawl_tasks[name] = task

        self._analytic_db: Union['AnalyticDataStore', None] = None
        self._crawl_db: Union['CrawledDataStore', None] = None

    def get_normalizer(self,
                       name: str = "default") -> Normalizer:
        return Normalizer(**self.normalizers[name])

    @property
    def language_processor(self) -> LanguageProcessor:
        return LanguageProcessor(self.language, self.token_matchers)

    def __repr__(self):
        return f"<Project(name='{self.name}', description='" \
               f"{self.description}', languages={self.language})>"

    @property
    def analytic_db(self) -> AnalyticDataStore:
        if self._analytic_db is None or not self._analytic_db.session.is_active:
            self._analytic_db = AnalyticDataStore(self.path / 'analytic.db')
        return self._analytic_db

    @property
    def crawl_db(self) -> CrawledDataStore:
        if self._crawl_db is None or not self._crawl_db.session.is_active:
            self._crawl_db = CrawledDataStore(self.path / 'crawl.db')
            for task in self.crawl_tasks.values():
                self._crawl_db.restore_query_metadata(task)
        return self._crawl_db

    def save(self):
        if self._analytic_db is not None:
            self._analytic_db.commit()
            self._analytic_db.close()
        if self._crawl_db is not None:
            self._crawl_db.commit()
            self._crawl_db.close()
