from .llms import *


class BechdelLLM(LLM):
    system_prompt = """You are a literary critic assessing whether scenes from a screenplay pass the "Bechdel test", which involves 3 components: 
1. Does the scene contain more than 1 female character?
2. If (1) is true, do at least two female characters speak to one another?
3. If (2) is true, do they speak about something other than a man?

Read the following story and return valid JSON dictionary containing this information.
"""


class BechdelScreenplayLLM(BechdelLLM):
    corpus_root = "/Users/ryan/github/bechdeltest/data/corpus"
    filename = "data.bechdel.db"

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

    @classmethod
    def parse_script(self, id, model="gemini-pro", force=False, progress=True):
        with self.get_db() as db:
            if not force and id in db and type(db[id]) == pd.DataFrame and len(db[id]):
                return db[id]

            o = []
            for scene_num, scene in tqdm(
                list(self.iter_scenes(id)), position=0, disable=not progress
            ):
                res = self.generate(scene, model=model, verbose=False)
                try:
                    resd = json.loads("{" + res.split("{", 1)[-1].split("}")[0] + "}")
                    resd["scene_num"] = scene_num
                    o.append(resd)
                except Exception as e:
                    pass

            try:
                df = pd.DataFrame(o).fillna("")
                df["num_female_characters"] = df["female_characters"].apply(
                    lambda x: len(x) if x else 0
                )
                df["num_male_characters"] = df["male_characters"].apply(
                    lambda x: len(x) if x else 0
                )
                for k in [
                    "female_characters_speak_to_each_other",
                    "female_characters_speak_about_something_other_than_a_man",
                ]:
                    df[k] = df[k].apply(lambda x: x == True)
            except Exception as e:
                df = pd.DataFrame()

            db[id] = df
            return df

    @classmethod
    def parse_all_scripts(self, ids=None, model="gemini-pro"):
        df = self.get_meta()
        df = df[~df.imdb_id.isna()]
        ids = df.id if not ids else ids
        for id in tqdm(ids, position=0):
            self.parse_script(id, progress=True, model=model)


    @classmethod
    def migrate_db_format(self, dbfn1='data.bechdel.db', dbfn2='data.bechdel.by_scene.db'):
        dbfnfn1=os.path.join(PATH_DATA,dbfn1)
        dbfnfn2=os.path.join(PATH_DATA,dbfn2)
        with SqliteDict(dbfnfn1, flag='r') as db1, SqliteDict(dbfnfn2, flag='c', autocommit=True) as db2:
            for id in tqdm(list(db1.keys())):
                df=db1[id]
                if type(df)!=pd.DataFrame or not len(df): continue
                for rowd in df.to_dict(orient='records'):
                    rowd['id']=id
                    id2=f'{id}__{rowd['scene_num']}'
                    db2[id2]=rowd


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
