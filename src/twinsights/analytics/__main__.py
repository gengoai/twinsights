import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from matplotlib import pyplot as plt
from tqdm import tqdm

from twinsights.analytics.clustering import Clustering
from twinsights.analytics.comatrix import CoMatrix
from twinsights.analytics.visualization import generate_wordcloud
from twinsights.app import App, Arg, Command
from twinsights.project import Project


class AnalyticsApp(App):
    __description__ = 'GengoAI Social Media Analytics App'
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

    @Command(help="Export")
    def export(self):
        with open(f"{self.project.name}.json", "w") as fp:
            lines = []
            with self.project.analytic_db as db:
                for post in db.get_posts():
                    t = re.sub("#[A-Za-z0-9_]+", "", post.text.content).strip()
                    if t != "":
                        lines.append({
                            "text": t,
                            "label": "Yes" if self.project.name == 'tw' else 'No'
                        })
            json.dump(lines, fp)

    @Command(help="Build commatrix")
    def build_comatrix(self,
                       name: str,
                       item1: str,
                       item2: str,
                       min_item1_count: int = 1,
                       min_item2_count: int = 1,
                       count_by: str = 'post',
                       scorer: str = 'tf',
                       normalizer="default"):
        with self.project.analytic_db as db:
            comatrix = CoMatrix(name=name,
                                db=db,
                                normalizer=self.project.get_normalizer(
                                    normalizer))
            comatrix.build(item1,
                           item2,
                           min_item1_count=min_item1_count,
                           min_item2_count=min_item2_count,
                           count_by=count_by,
                           scorer=scorer)

    @Command(help="Update all comatrices")
    def update_comatrices(self):
        for name, parameters in self.project.comatrices.items():
            print(f"Updating {name}")
            parameters["name"] = name
            self.build_comatrix(**parameters)

    @Command(help="Cluster a comatrix")
    def cluster(self,
                comatrix: str,
                k: int,
                normalize: bool = False):
        clusterer = Clustering(self.project, f"{comatrix}_clusters")
        with self.project.analytic_db as db:
            clusterer.cluster(CoMatrix.load(comatrix, db), k, normalize)
            clusterer.save(db)

    @Command(help="Cluster and visualization multiple K")
    def multik_cluster(self,
                       comatrix: str,
                       min_k: int = 2,
                       max_k: int = 20,
                       normalize: bool = False):
        output_dir = self.project.path / comatrix
        if output_dir.exists():
            from shutil import rmtree
            rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for k in tqdm(range(min_k, max_k + 1)):
            self.cluster(comatrix, k, normalize)
            with self.project.analytic_db as db:
                clusterer = Clustering.load(comatrix, self.project, db)
                cm = CoMatrix.load(comatrix, db)
                fig = clusterer.visualize(cm)
                output_path = output_dir / f"{k}.png"
                fig.savefig(output_path)
                fig = clusterer.tag_cloud([cm])
                output_path = output_dir / f"{k}-tagcloud.png"
                fig.savefig(output_path)

    @Command(
        help="Update Clusters for each CoMatrix automatically determining K")
    def update_clusters(self):
        for name, parameters in self.project.cluster_parameters.items():
            print(f"Updating {name}_clusters")
            parameters["comatrix"] = name
            self.cluster(**parameters)

    @Command(help="Cluster a comatrix")
    def cluster_viz(self,
                    comatrix: str):
        with self.project.analytic_db as db:
            clusterer = Clustering.load(comatrix, self.project, db)
            cm = CoMatrix.load(comatrix, db)
            fig = clusterer.visualize(cm)
            return fig

    @Command(help="Build a wordcloud over tokens")
    def wordcloud(self,
                  word: str = "token",
                  count_by: str = "post",
                  normalizer: str = "default"):
        word = word.lower()
        count_by = count_by.lower()
        n = self.project.get_normalizer(normalizer)
        counts = defaultdict(float)
        with self.project.analytic_db as db:
            if word in ("token", "entity", "chunk"):
                items = db.get_spans(word, count_by, n)
            elif word == "hashtag":
                items = db.get_hashtags(count_by)
            elif word == "empath":
                items = db.get_empath(count_by)
            else:
                raise ValueError(f"{word} is invalid")

            for item in items:
                if item is None:
                    continue
                for w in set(item):
                    counts[w] += 1

        generate_wordcloud(counts,
                           title=f"{word.title()} WordCloud for "
                                 f"{count_by.title()}s")
        plt.show()

    @Command(help="Word cloud visualization of clusters",
             args=[Arg(dest="cnames", nargs="+",
                       type=lambda x: x)])
    def cluster_wordcloud(self,
                          name: str,
                          file: str,
                          cnames: List[str] = None):
        with self.project.analytic_db as db:
            comatrix_names = [] if cnames is None else cnames
            comatrix_names.append(name)
            comatrices = [CoMatrix.load(cname, db) for cname in comatrix_names]
            clusters = Clustering.load(name, self.project, db)
            fig = clusters.tag_cloud(comatrices)
            fig.savefig(file)

    @Command(
        help="Performs ETL from crawl database creating an analytics database")
    def etl(self):
        lp = self.project.language_processor
        with self.project.crawl_db as crawl_db:
            with self.project.analytic_db as analytic_db:
                for datum in tqdm(crawl_db.get_data(),
                                  total=crawl_db.status_count()):
                    etl_processor = datum.api.etl()
                    for user, post in etl_processor(datum, crawl_db, lp):
                        analytic_db.add_or_update(user)
                        analytic_db.commit()
                        analytic_db.add_or_update(post)
                        analytic_db.commit()
        for name, parameters in self.project.comatrices.items():
            parameters["name"] = name
            self.build_comatrix(**parameters)

    @Command(help="")
    def hashtags(self):
        with self.project.analytic_db as db:
            freqs = db.hashtag_frequencies()
            for tag, count in sorted(freqs.items(), key=lambda x: (x[1], x[0]),
                                     reverse=True):
                if count > 20:
                    print(tag, count)


if __name__ == '__main__':
    AnalyticsApp().run()
