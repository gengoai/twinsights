import datetime
from pathlib import Path
from typing import Optional

from twinsights.app import App, Arg, Command
from twinsights.crawler.twitter import TwitterUserCrawler
from twinsights.project import Project


class CrawlerApp(App):
    __description__ = 'GengoAI Social Media Crawling and Importing'
    __version__ = "1.0"

    def __init__(self):
        super().__init__()
        self.project: Optional[Project] = None
        self.add_global_argument(Arg(
            dest='project',
            option_names=['-p', '--project'],
            help="The path to the project to use",
            type=lambda path: Project(Path(path)),
            required=True
        ))

    def setup(self) -> None:
        self.project = self.parsed_global_args["project"]

    @Command(name="tweets",
             help="Perform a Crawl")
    def crawl(self,
              historical: bool = False):
        with self.project.crawl_db as db:
            # Initialize the crawlers and restore any persistent settings
            # about the crawl from the database namely, the max_id and since_id
            crawlers = [task.api.crawler(task) for task in
                        self.project.crawl_tasks.values()]
            # Perform the crawl
            next_step = []
            while len(crawlers) > 0:
                for crawler in crawlers:
                    crawler.historical = historical
                    if crawler.in_timeout():
                        # If a crawler is in timeout, we will skip
                        # it until it can try and call again
                        next_step.append(crawler)
                    elif crawler.crawl(db):
                        # A crawler will return False when it has
                        # exhausted the data it can crawl
                        next_step.append(crawler)
                        crawler.last_run = datetime.datetime.now()

                crawlers = next_step.copy()
                next_step = []
                db.commit()

    @Command(name="users", help="User Timeline Crawler")
    def user_timeline(self):
        user_crawler = TwitterUserCrawler(self.project.language)
        with self.project.crawl_db as db:
            user_crawler.crawl(db)


if __name__ == '__main__':
    CrawlerApp().run()
