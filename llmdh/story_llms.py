from .llms import *


class BechdelLLM(LLM):
    system_prompt = """You are a literary critic assessing whether scenes from a screenplay pass the "Bechdel test", which involves 3 components: 
1. Does the scene contain more than 1 female character?
2. If (1) is true, do at least two female characters speak to one another?
3. If (2) is true, do they speak about something other than a man?

Read the following story and return valid JSON dictionary containing this information.
"""

    @classmethod
    def parse(self, scene_txt, model="gemini-pro"):
        res = self.generate(scene_txt, model=model, verbose=False)
        try:
            resd = json.loads("{" + res.split("{", 1)[-1].split("}")[0] + "}")
            resd["num_female_characters"] = len(resd["female_characters"])
            resd["num_male_characters"] = len(resd["male_characters"])
            for k in [
                "female_characters_speak_to_each_other",
                "female_characters_speak_about_something_other_than_a_man",
            ]:
                resd[k] = resd[k] == True
            return resd
        except Exception as e:
            return None

    @classmethod
    def parse_all(self, model="gemini-pro"):
        with self.get_corpus_db() as dbc, self.get_db() as db:
            todo = list(set(dbc.keys()) - set(db.keys()))
            random.shuffle(todo)
            for id_scene in tqdm(todo):
                resd = self.parse(dbc[id_scene], model=model)
                if resd is not None:
                    resd["id"], resd["scene_num"] = id_scene.split("__", 1)
                    resd["scene_num"] = int(resd["scene_num"])
                    db[id_scene] = resd


class BechdelScreenplayLLM(BechdelLLM):
    corpus_root = "/Users/ryan/github/bechdeltest/data/corpus"
    filename = "data.bechdel.by_scene.db"

    example_prompts = [
        (
            """
            [INT. HOUSE]

    EMILY: I don't know, Emma, what do you think?

    EMMA: I don't know either! Michael seems too risky to date to me.

    EMILY: But how do you know?

    EMMA: Well, I—

            (Just then MICHAEL enters through the back door.)

    MICHAEL: What are you guys talking about?

            (EMILY and EMMA blush.)

    EMMA: Nothing!

    EMILY: Well, almost nothing.
    """,
            """
    {
        "female_characters": ["Emily", "Emma"],
        "male_characters": ["Michael"],
        "female_characters_speak_to_each_other": true,
        "female_characters_speak_about_something_other_than_a_man": false,
        "explanation": "Emily and Emma, both women, speak to each other, but their conversation is limited that regarding the reputation of Michael, a man."
    }
    """,
        )
    ]

    @classmethod
    @cache
    def get_meta(self):
        dfmeta = pd.read_csv(self.corpus_root + "/metadata.csv")
        dfmeta["release_date"] = pd.to_numeric(dfmeta["release_date"], errors="coerce")
        dfmeta["dec"] = dfmeta["release_date"] // 10 * 10
        return dfmeta.sample(frac=1)

    @classmethod
    def iter_scenes(self, id):
        fn = os.path.join(self.corpus_root, "texts", id, f"{id}.dialogue.csv")
        if os.path.exists(fn):
            df = pd.read_csv(fn).fillna("")

            o = []
            for g, gdf in df.groupby("scene_num"):
                scene = []
                for i, row in gdf.iterrows():
                    if not scene and row.scene_desc:
                        scene.append(f"\t\t[{row.scene_desc}]")
                    if row.narration:
                        scene.append(f"\t\t({row.narration})")
                    line = row.speaker
                    if row.direction:
                        line += f" ({row.direction})"
                    line += f": {row.speech}"
                    scene.append(line)
                yield g, ("\n\n".join(scene))

    # @classmethod
    # def parse_script(self, id, model="gemini-pro", force=False, progress=True):
    #     with self.get_db() as db:
    #         if not force and id in db and type(db[id]) == pd.DataFrame and len(db[id]):
    #             return db[id]

    #         o = []
    #         for scene_num, scene in tqdm(
    #             list(self.iter_scenes(id)), position=0, disable=not progress
    #         ):
    #             res = self.generate(scene, model=model, verbose=False)
    #             try:
    #                 resd = json.loads("{" + res.split("{", 1)[-1].split("}")[0] + "}")
    #                 resd["scene_num"] = scene_num
    #                 o.append(resd)
    #             except Exception as e:
    #                 pass

    #         try:
    #             df = pd.DataFrame(o).fillna("")
    #             df["num_female_characters"] = df["female_characters"].apply(
    #                 lambda x: len(x) if x else 0
    #             )
    #             df["num_male_characters"] = df["male_characters"].apply(
    #                 lambda x: len(x) if x else 0
    #             )
    #             for k in [
    #                 "female_characters_speak_to_each_other",
    #                 "female_characters_speak_about_something_other_than_a_man",
    #             ]:
    #                 df[k] = df[k].apply(lambda x: x == True)
    #         except Exception as e:
    #             df = pd.DataFrame()

    #         db[id] = df
    #         return df

    # @classmethod
    # def parse_all_scripts(self, ids=None, model="gemini-pro"):
    #     df = self.get_meta()
    #     df = df[~df.imdb_id.isna()]
    #     ids = df.id if not ids else ids
    #     for id in tqdm(ids, position=0):
    #         self.parse_script(id, progress=True, model=model)

    # @classmethod
    # def migrate_db_format(self, dbfn1='data.bechdel.db', dbfn2='data.bechdel.by_scene.db'):
    #     dbfnfn1=os.path.join(PATH_DATA,dbfn1)
    #     dbfnfn2=os.path.join(PATH_DATA,dbfn2)
    #     with SqliteDict(dbfnfn1, flag='r') as db1, SqliteDict(dbfnfn2, flag='c', autocommit=True) as db2:
    #         for id in tqdm(list(db1.keys())):
    #             df=db1[id]
    #             if type(df)!=pd.DataFrame or not len(df): continue
    #             for rowd in df.to_dict(orient='records'):
    #                 rowd['id']=id
    #                 id2=f'{id}__{rowd['scene_num']}'
    #                 db2[id2]=rowd

    @classmethod
    def get_corpus_db(self, dbfn="data.bechdel.corpus.db"):
        dbfnfn = os.path.join(PATH_DATA, dbfn)
        return SqliteDict(dbfnfn, flag="c", autocommit=True)

    @classmethod
    def prepare_corpus(self):
        with self.get_corpus_db() as db:
            for id in tqdm(self.get_meta().id):
                for scene_num, scene_txt in self.iter_scenes(id):
                    id2 = f"{id}__{scene_num}"
                    db[id2] = scene_txt

    @classmethod
    def get_figdf(self):
        import imdb

        dfmeta = self.get_meta()
        db = self.get_db()
        keys = [k for k in db.keys() if type(db[k]) is pd.DataFrame and len(db[k])]
        sumdf = (
            pd.DataFrame([db[k].mean(numeric_only=True) for k in keys], index=keys)
            .rename_axis("id")
            .dropna()
        )
        dfall = sumdf.merge(dfmeta, on="id")
        dfall["release_date"] = pd.to_numeric(dfall["release_date"], errors="coerce")

        # get genres
        ia = imdb.IMDb()
        imdb_ids = dfall.imdb_id.dropna().apply(int).apply(str)
        dbgenre = SqliteDict(
            os.path.join(PATH_DATA, "data.imdb_genre.db"), flag="c", autocommit=True
        )
        for idx in tqdm(imdb_ids):
            if idx not in dbgenre:
                dbgenre[idx] = ia.get_movie(idx)
            # time.sleep(0.005)

        def tostr(x):
            try:
                return str(int(x))
            except:
                return ""

        dfall["imdb_id"] = dfall.imdb_id.apply(tostr)

        dfall2 = pd.DataFrame(
            {**d, "genre": g}
            for d in dfall.reset_index().to_dict("records")
            for g in dbgenre.get(d["imdb_id"], {}).get("genres", [])
            if g not in {"Music", "Short"}
        ).set_index("id")

        return dfall2

    @classmethod
    def plot_boxplot(
        self, y="female_characters_speak_about_something_other_than_a_man"
    ):
        dfall2 = self.get_figdf()
        genres = dfall2.genre.value_counts()
        genres = genres[genres >= 20].index
        genreorder = (
            dfall2.groupby("genre")[y].median().sort_values(ascending=False).index
        )
        dfall2["genre"] = pd.Categorical(dfall2["genre"], categories=genreorder)
        figdf = dfall2[dfall2.genre.isin(genres)]
        genreorderl = [g for g in genreorder if g in genres]
        fig = p9.ggplot(figdf[figdf.genre != ""], p9.aes(x="genre", y=y))
        fig += p9.geom_boxplot(outlier_shape="")
        fig += p9.geom_point(shape="x")
        fig += p9.coord_flip()
        fig += p9.scale_x_discrete(limits=genreorderl)
        # fig+=p9.theme_minimal()
        fig += p9.labs(
            x="Primary genre of film",
            y=f'% scenes in which {y.replace("_"," ")}',
            title="Comparing genres for Bechdel test",
        )
        return fig

    @classmethod
    def plot_genre_biplot(
        self,
        x="female_characters_speak_to_each_other",
        y="female_characters_speak_about_something_other_than_a_man",
    ):
        dfall2 = self.get_figdf()
        figdf = pd.concat(gdf for g, gdf in dfall2.groupby("genre") if len(gdf) >= 20)
        figdf[x] *= 100
        figdf[y] *= 100
        dfgenres = figdf.groupby("genre").median(numeric_only=True)
        dfgenres["count"] = figdf.genre.value_counts()
        fig = p9.ggplot(dfgenres.reset_index(), p9.aes(x=x, y=y, label="genre"))
        fig += p9.geom_smooth(method="lm", color="blue", alpha=0.25, size=0.5)
        # fig += p9.geom_point(p9.aes(size='count'), alpha=.25)
        fig += p9.geom_text()
        # fig += p9.geom_abline(intercept=0, slope=1, color='blue')
        # fig += p9.scale_x_continuous(limits=(0,25))
        # fig += p9.scale_y_continuous(limits=(0,12.5))
        fig += p9.theme_minimal()
        fig += p9.labs(
            x=f'% scenes in which {x.replace("_", " ")}',
            y=f'% scenes in which {y.replace("_", " ")}',
            title="Bechdel test components by genre",
        )
        fig.save("fig.genre_biplot.png")
        return fig


class NovelLLM(LLM):
    corpus_root = "/Users/ryan/lltk_data/corpora/canon_fiction/txt"
    corpus_meta = "/Users/ryan/lltk_data/corpora/canon_fiction/metadata.xls"

    def __init__(self, filename, *args, **kwargs):
        kwargs["name"] = filename
        fnfn = os.path.join(self.corpus_root, filename)
        if not os.path.exists(fnfn):
            raise Exception("file not found")
        with open(fnfn) as f:
            kwargs["user_prompt"] = f.read().strip()
        super().__init__(*args, **kwargs)

    @classmethod
    @property
    def meta(self):
        return pd.read_excel(self.corpus_meta).set_index("id")

    @classmethod
    @property
    def canon_filenames(self):
        df = self.meta.fillna("")
        return [idx + ".txt" for idx in df[df.canon_genre != ""].index]

    @cached_property
    def parsed_response(self):
        res = self.response
        try:
            return dirty_jsonl_loads(res)
        except Exception as e:
            logger.error(f"First failed: {e}")
            if res:  # and "{" in res and "}" in res:
                res = self.generate_gemini(
                    system_prompt="Reformat this into valid JSON. Return ONLY the json.",
                    user_prompt=self.response,
                    model="gemini-1.5-flash-latest",
                )
                try:
                    data = dirty_jsonl_loads(res)
                    self.__dict__["response"] = res
                    logger.warning("fixed by second request")
                    return data
                except Exception as e:
                    logger.error(f"Reparse failed: {e}")
                    return None
            else:
                return None

    @classmethod
    def gen_all(self, force=False, **kwargs):
        done = set(self.db.keys()) if not False else set()
        # poss = set([fn for fn in os.listdir(self.corpus_root) if fn.endswith(".txt")])
        poss = {fn for fn in self.canon_filenames}
        todo = list(poss - done)
        random.shuffle(todo)
        with logmap("parsing all novels") as lm:
            for fn in lm.iter_progress(todo):
                llm = self(filename=fn, force=force, **kwargs)
                llm.save()
                nap = random.randrange(5, 25)
                if llm.result:
                    lm.set_progress_desc(
                        f"found {len(llm.result)} in {fn}. sleeping for {nap}s"
                    )
                else:
                    lm.set_progress_desc(f"failed on {fn}. sleeping for {nap}s")
                time.sleep(nap)

    def graph(self):
        import networkx as nx

        G = nx.Graph()
        for d in self.result:
            if not d.get("character1_name") or not d.get("character2_name"):
                continue
            for ck in ["character1_name", "character2_name"]:
                d[ck] = d[ck].split("(")[0].replace("&wblank;", "").strip()
            for n in range(2):
                pref = f"character{n+1}_"
                name = d[pref + "name"]
                if not G.has_node(name):
                    nd = {k.replace(pref, ""): d[k] for k in d if k.startswith(pref)}
                    G.add_node(name, **nd)
            G.add_edge(
                d["character1_name"],
                d["character2_name"],
                **{k: d[k] for k in d if not k.startswith("character")},
            )
        return G


class NovelRelationshipsLLM(NovelLLM):
    filename = "data.novel_relationships2.canon_fiction.db"

    system_prompt = """
You are a literary critic. Your task is to extract and describe all ROMANTIC relationships in a novel, whether actual or suggested as potential. 

For each actual or potential romantic relationship between characters, determine:
1) The name, gender, class, annual income (if explicitly stated), and a description of both characters involved in the relationship
2) Whether the characters were already in a relationship at the start of the novel
3) Whether the characters did indeed enter into an actual relationship (not merely flirtation) at any point in the novel
3) Whether the characters end up still in a relationship by the end of the novel
5) A numerical estimate of the likelihood of a relationship forming, in the range of 0 to 10. For example, if two matches for a character are suggested as potential, and one of the is a dashing Mr Truelove and another a contemptible Mr Fool, the reader can suspect that Mr Trueworth will be the eventual match (likelihood 9), and not Mr Fool (likelihood 1)
6) The nature of the eventual relationship (whether marriage or unmarried partnership)
7) If the relationship ends, determine the manner in which it ended (e.g. divorce, death of one or both characters, etc)
8) A short summary of the relationship's beginning, middle, and end

Return a list of JSON dictionaries of the following form:

[
    {
        "character1_name": "Elizabeth Bennet",
        "character2_name": "Mr Darcy",
        "character1_gender": "Female",
        "character2_gender": "Male",
        "character1_class": "middle gentry",
        "character2_class": "upper gentry",
        "character1_income": "£2,000 (her father, Mr Bennet's income)",
        "character2_income": "£10,000",
        "character1_desc": "Elizabeth Bennet is the intelligent and spirited second daughter of a moderately wealthy landed gentleman.",
        "character2_desc": "Mr Darcy is a wealthy and initially aloof gentleman of high social standing.",
        "relationship_at_start": false,
        "relationship_happened": true,
        "relationship_by_end": true,
        "relationship_likelihood": 7,
        "relationship_is_marriage": true,
        "relationship_ended": false,
        "relationship_ended_reason": "",
        "relationship_summary": "Elizabeth meets Mr Darcy at a ball, where she overhears him insulting him; Elizabeth harbors contempt for him, and rejects his first offer of marriage; but she warms to him when he demonstrates his selfless love by helping her resolve a scandal involving her sister, Lydia; she accepts his second, humbler of marriage."
    },
    
]
"""

    def plot_network(self):
        import networkx as nx
        import matplotlib.pyplot as plt

        G = self.graph()

        # plt.figure(figsize=(8, 6))
        fig, ax = plt.subplots(figsize=(9, 9))

        def get_color(d):
            if not "gender" in d:
                return "gray"
            if "Male" in d["gender"].title():
                return "lightblue"
            if "Female" in d["gender"].title():
                return "bisque"
            return "gray"

        # Get node colors based on gender attribute
        node_colors = [get_color(d) for node, d in G.nodes(data=True)]

        # Get edge widths based on relationship_likelihood attribute
        def get_size(d):
            try:
                return int(d["relationship_likelihood"])
            except Exception:
                return 1

        edge_widths = [get_size(d) for a, b, d in G.edges(data=True)]

        # Define color mapping for 'relationship_happened'
        edge_color_map = {
            "true": "seagreen",
            "True": "seagreen",
            "Presumed True": "seagreen",
            True: "seagreen",
            False: "orangered",
            "false": "orangered",
            "False": "orangered",
        }
        edge_colors = [
            (edge_color_map[G.edges[edge]["relationship_happened"]]) for edge in G.edges
        ]

        nx.draw(
            G,
            with_labels=True,
            node_color=node_colors,
            node_size=500,
            edge_color=edge_colors,
            width=edge_widths,
            font_size=12,
            font_color="black",
        )
        if self.name.startswith("chadwyck."):
            tdat = self.name.split(".")
            tname = f'{tdat[1]}, {tdat[2].replace("_"," ").upper()}'
        else:
            tname = self.name.replace(".txt", "")

        ax.set_title(f"Relationships in {tname}")
        plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make space for the caption
        fig.text(
            x=0,  # Starting from the very left of the axes
            y=0,  # A small vertical offset from the bottom
            s="Legend: Green edges are actualized relationships; red edges are relationships suggested but unactualized.\nEdge width indicates likelihood of relationship forming, assessed by the LLM.\nWomen are shown in orange, men in blue.",
            ha="left",  # Left horizontal alignment
            va="bottom",  # Bottom vertical alignment
            transform=ax.transAxes,  # Use axes coordinate system
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                PATH_DATA,
                "relationships",
                f'relationship_graphs.{self.name.replace(".txt",".png")}',
            ),
            dpi=300,
        )
        plt.close()
        # plt.show()
        return plt


class NovelCharactersLLM(NovelLLM):
    corpus_root = "/Users/ryan/lltk_data/corpora/canon_fiction/txt"
    filename = "data.novel_characters.canon_fiction.db"

    system_prompt = """
You are a literary critic. Your task is to extract and describe all characters in a novel, major or minor, named or unnamed. 

For each character, determine:
1) The full name of the character, along with any other names by which they are referred
2) The gender of the character
3) The socioeconomic class of the character
4) The annual income of the character (if explicitly stated)
5) A numerical score (between 1 and 10) indicating the relative prominence of the character to the narrative
6) A description of the character's role in the novel
7) A description of the character's appearance in the novel
8) Copy the paragraph(s) in which the character first appears
9) Copy all passages that describe the character
10) Copy all passages in which the character performs a significant action (or any action, if a minor character)

Return a list of JSON dictionaries of the following form:

[
    {
        "fullname": "Fitzwilliam Darcy",
        "names": ["Mr Darcy", "Mr Fitzwilliam Darcy"],
        "gender": "Male",
        "class": "upper gentry",
        "income": "£10,000",
        "prominence": 9,
        "desc_role": "Mr Darcy is a wealthy and initially aloof gentleman of high social standing.",
        "desc_appearance": "A fine, tall person, with handsome features",
        "passage_first_appearance": "Mr. Bingley was good-looking and gentlemanlike; he had a pleasant countenance, and easy, unaffected manners. His sisters were fine women, with an air of decided fashion. His brother-in-law, Mr. Hurst, merely looked the gentleman; but his friend Mr. Darcy soon drew the attention of the room by his fine, tall person, handsome features, noble mien; and the report which was in general circulation within five minutes after his entrance, of his having ten thousand a year. The gentlemen pronounced him to be a fine figure of a man, the ladies declared he was much handsomer than Mr. Bingley, and he was looked at with great admiration for about half the evening, till his manners gave a disgust which turned the tide of his popularity; for he was discovered to be proud; to be above his company, and above being pleased; and not all his large estate in Derbyshire could then save him from having a most forbidding, disagreeable countenance, and being unworthy to be compared with his friend.",
        "passages_of_description": {
            "First Appearance": "Mr. Darcy soon drew the attention of the room by his fine, tall person, handsome features, noble mien; and the report which was in general circulation within five minutes after his entrance, of his having ten thousand a year.",
            "Initial Observations at the Meryton Ball": "The gentlemen pronounced him to be a fine figure of a man, the ladies declared he was much handsomer than Mr. Bingley, and he was looked at with great admiration for about half the evening, till his manners gave a disgust which turned the tide of his popularity.",
            "Elizabeth's Reflection": "Elizabeth found that he was as handsome as Mr. Bingley, but more so in the way of a manly, dignified appearance.",
            "Re-encounter at Pemberley": "Elizabeth, as they drove along, watched for the first appearance of Pemberley Woods with some perturbation; and when at length they turned in at the lodge, her spirits were in a high flutter. She had seen nothing of Bingley since the day of the dance, but she knew that his sisters were at home; and there was scarcely an alteration in their reception, the same civilities; the same enquiries; and very nearly the same complaints as before. It was his twin brother—Mr. Darcy himself—who astonished her. Not all that she had seen of him before had prepared her for the change in his manners. The report of his having ten thousand a year again became the top topic, and every eye was turned towards them."
        },
        "passages_of_action": {
            "Darcy's First Proposal": "In vain I have struggled. It will not do. My feelings will not be repressed. You must allow me to tell you how ardently I admire and love you.",
            "Darcy's Letter to Elizabeth": "Be not alarmed, madam, on receiving this letter, by the apprehension of its containing any repetition of those sentiments or renewal of those offers which were last night so disgusting to you.",
            "Darcy's Intervention in Lydia's Elopement": "Mr. Darcy called on Mr. Gardiner, and on being admitted to his conference, told him that he had some important business to communicate, in which his interest was most particularly concerned. “The matter which brings me so immediately to town,” continued Mr. Darcy, after sitting down in a chair, “is of a most serious nature, and I hope it will be productive of good to all concerned. When my brother-in-law left your house yesterday, he left it, I am sure, with the strongest convictions that it was improper for me to be trusted with such a secret, yet the necessity of concealment has been obviated, and I can now have no scruple in relating the whole to you. For the last fortnight I have been here, working on behalf of Miss Lydia Bennet. To Mr. Wickham himself I have been twice; I have sought a reconciliation with him on his first removal from Meryton, and then on his being next in town.",
            "Darcy's Second Proposal": "You are too generous to trifle with me. If your feelings are still what they were last April, tell me so at once. My affections and wishes are unchanged, but one word from you will silence me on this subject for ever.",
            "Darcy and Elizabeth at Pemberley": "They were within twenty yards of each other, and so abrupt was his appearance, that it was impossible to avoid his sight. Their eyes instantly met, and the cheeks of each were overspread with the deepest blush. He absolutely started, and for a moment seemed immovable from surprise; but shortly recovering himself, advanced towards the party, and spoke to Elizabeth, if not in terms of perfect composure, at least of perfect civility."
        }
    },
    
]
"""
