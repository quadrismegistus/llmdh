import sys

sys.path.insert(0, "/Users/ryan/github/logmap")
sys.path.insert(0, "/Users/ryan/github/prosodic")
from diskcache import Cache
from typing import *
import numpy as np
import re
from io import StringIO
import time
import calendar
from base64 import b64decode, b64encode
import datetime as dt
from functools import lru_cache, cached_property
from contextlib import contextmanager
from pprint import pprint, pformat
import prosodic
from tqdm import tqdm

tqdm.pandas()

cache = lru_cache(maxsize=None)
import pandas as pd
from collections import defaultdict
from sqlitedict import SqliteDict
import os
from logmap import logmap, logger
import warnings

warnings.filterwarnings("ignore")
import random
import json
from urllib.parse import quote as urlquote
import pickle

from google.generativeai.types import (
    HarmCategory,
    HarmBlockThreshold,
    GenerationConfig,
)
import ollama
import google.generativeai as genai
from openai import OpenAI
import plotnine as p9

pd.options.display.max_rows = 100

p9.options.figure_size = 8, 7
pd.options.display.max_columns = None

# paths
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_HOME = os.path.expanduser("~/llmdh_data")
PATH_REPO = os.path.dirname(PATH_HERE)
# PATH_DATA = os.path.join(PATH_HOME, "data")
PATH_DATA = PATH_HOME
os.makedirs(PATH_DATA, exist_ok=True)


LLM_DEFAULT_MODEL_LOCAL = "llama2-uncensored:7b"
LLM_DEFAULT_MODEL_REMOTE = "gemini-1.5-pro-latest"
LLM_DEFAULT_MODEL_OPENAI = "gpt-3.5-turbo"
LLM_DEFAULT_MODEL_CLAUDE = "claude-3-haiku-20240307"
LLM_DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL", LLM_DEFAULT_MODEL_REMOTE)
OPENAI_BASEURL = None

MAX_TOKENS = 1000
MODEL_NICKNAMES = {
    "gemini": "gemini-pro",
    "hermes": "nous-hermes-llama2-13b",
    "mistral": "mistral-7b-instruct",
    "orca": "orca-2-7b",
    "gpt3turbo": "gpt-3.5-turbo",
    "gpt4turbo": "gpt-4-turbo",
    "gpt4": "gpt-4",
}
PATH_KAGGLE = os.path.join(PATH_DATA, "kaggle_poem_dataset.csv")
PATH_OUTPUT = os.path.join(PATH_DATA, "rhyme_analysis2.db")


cache_obj_rhyme = Cache(os.path.join(PATH_DATA, "cache.memoized.rhyme.dc"))
cache_obj_meter = Cache(os.path.join(PATH_DATA, "cache.memoized.meter.dc"))
DEFAULT_TEMP = 1.0

from .utils import *
