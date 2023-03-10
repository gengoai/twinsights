from typing import Any, Dict

Config: Dict[str, Any] = {
    "Api.TwitterStatus.ETL":
                    "twinsights.analytics.etl.TwitterStatusETL",
    "Api.TwitterStatus.Crawler":
                    "twinsights.crawler.twitter.TweepyCrawler",

    "spacy.models": {
        "en": "en_core_web_sm"
    }

}
