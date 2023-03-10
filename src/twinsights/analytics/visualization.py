import re
from typing import List, Callable, Union, Dict

import matplotlib.pyplot as plt
import pandas as pd

from twinsights.analytics.data import Post, User


def clean_text(text):
    """
    Args:
        text:
    """
    return re.sub(r'(http\S+|[@#]\S+|<[^>]+>|&\w+;)', '', text).lower().strip()


def generate_wordcloud(df: Union[pd.DataFrame, Dict[str, float], List[Union[Post, User]]],
                       text_column: str = None,
                       frequency_column: str = None,
                       width: int = 800,
                       height: int = 600,
                       colormap: str = 'Blues',
                       title: str = None,
                       axis_off=True,
                       cleaning_func: Callable[[str], str] = clean_text,
                       ax: plt.Axes = None):
    import wordcloud as wc

    wordcloud = wc.WordCloud(min_word_length=3,
                             width=width,
                             height=height,
                             normalize_plurals=True,
                             colormap=colormap)

    if isinstance(df, Dict):
        wordcloud = wordcloud.generate_from_frequencies(df)
    elif isinstance(df, pd.DataFrame):
        pdd = {k: float(v) for k, v in zip(df[text_column].values, df[frequency_column].values)}
        wordcloud = wordcloud.generate_from_frequencies(pdd)
    else:
        if isinstance(df[0], Post):
            text = "\n".join([cleaning_func(post.text) for post in df if post.text is not None])
        else:
            text = "\n".join([cleaning_func(post.description) for post in df if post.description is not None])
        wordcloud = wordcloud.generate(text)

    if ax is None:
        plt.imshow(wordcloud)
        if axis_off:
            plt.axis("off")
        if title is not None:
            plt.title(title)
    else:
        ax.imshow(wordcloud, interpolation='bilinear')
        if axis_off:
            ax.axis("off")
        if title is not None:
            ax.set_title(title, verticalalignment="top")
