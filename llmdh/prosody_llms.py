from .llms import *


class ProsodyLLM(LLM):
    # system_prompt = "You are an expert on rhyme and meter. Please generate a text according to the formal rules specified by the user."
    system_prompt = (
        "Please generate a text according to the formal rules specified by the user."
    )


class RhymeLLM(ProsodyLLM):
    # user_prompt = "Please write a poem that does NOT rhyme."
    system_prompt = "You are a poet who can follow formal rules and instructions."
    user_prompt = "Please write a poem with 14 lines, that does NOT rhyme. I REPEAT: DO NOT RHYME. LINES SHOULD NEVER RHYME."
    filekey = "RhymeLLM"
    models = ["gemini-pro", "llama2-uncensored:7b", "gpt-3.5-turbo"]
    # user_prompt = "Please write an UNRHYMED poem, i.e. a poem that does NOT rhyme. Ensure that all lines do NOT rhyme."
    # user_prompt = "Please write an UNRHYMED text, i.e. a multi-lined text whose lines do NOT rhyme. Ensure that all lines do NOT rhyme."
    # user_prompt = "Please write 10+ lines of text on any topic, with no more than 5-10 words on a single line. Separate lines by line breaks."

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.user_prompt += f"\n\nRandom seed: {random.random()}"
    #     self.system_prompt += f"\n\nRandom seed: {random.random()}"

    @cached_property
    def parsed_response(self):
        import prosodic

        poem_txt = self.response
        logger.warning(poem_txt)
        text = prosodic.Text(poem_txt)
        return {
            "poem": poem_txt,
            "num_stanzas": len(text.stanzas),
            "num_lines": len(text.lines),
            "num_rhymes": len(text.get_rhyming_lines()),
            "is_rhyming": text.is_rhyming,
        }

    @classmethod
    def gather_kaggle(self):
        with SqliteDict(PATH_OUTPUT, flag="r") as db:
            return pd.DataFrame(db[k] for k in db.keys())

    @classmethod
    @cache
    def gather_all(self, max_per=100, qstr="4<=num_lines<=14"):
        with logmap("gathering all") as lm, lm.verbosity(0):
            df1 = self.gather().query(qstr)  # .sample(frac=1)
            df1 = pd.concat(
                gdf.iloc[:max_per]  # if len(gdf) > max_per else gdf
                for g, gdf in df1.groupby(["model", "num_lines"])
            )

            df2 = self.gather_kaggle().query(qstr)
            df2 = pd.concat(
                gdf.sample(max_per) if len(gdf) > max_per else gdf
                for g, gdf in df2.groupby("num_lines")
            )
            return pd.concat(
                [
                    df1.assign(corpus="LLM", author="LLM"),
                    df2.assign(corpus="poetryfoundation.org", author="Poets"),
                ]
            ).fillna("")

    @classmethod
    def gather_parse(self, max_per=100, max_rhyme_dist=0):
        df = RhymeLLM.gather_all(max_per=max_per)
        df["poem_txt"] = [x or y for x, y in zip(df.poem, df.Content)]
        df = df.drop_duplicates("poem_txt")
        o = []
        with logmap("parsing") as lm:
            for i, row in lm.iter_progress(df.iterrows(), total=len(df)):
                try:
                    with lm.verbosity(0):
                        res = parse_rhyme_txt(
                            row.poem_txt,
                            max_rhyme_dist=max_rhyme_dist,
                        )
                    odx = {
                        **dict(row),
                        **res,
                    }
                    o.append(odx)

                except Exception as e:
                    lm.error(e)
        odf = pd.DataFrame(o)
        return odf


class SonnetLLM(ProsodyLLM):
    system_prompt = (
        "Please generate a text according to the formal rules specified by the user."
    )
    user_prompt = "Write 1 new sonnet, imitating exactly the style and form of Shakespeare's sonnets. Use line breaks (\n) to separate lines. Your sonnet MUST have 14 lines."
    filekey = "SonnetLLM"
    models = ["gemini-pro", "llama2-uncensored:7b"]

    @classmethod
    @cache
    def gather_parse(self, lim=None):
        with logmap("gathering and parsing") as lm:

            # shak
            dfshak = self.gather_parse_shakespeare().iloc[:lim]
            if lim is None:
                lim = len(dfshak) * 1

            o = []
            df = self.gather().query('model=="gemini-pro"')
            for g, gdf in df.groupby("model"):
                og = []
                gdf = gdf.sort_index()
                for i, row in lm(
                    gdf.iterrows(),
                    desc=f"parsing llm results for model {g}",
                    total=len(gdf),
                ):
                    if lim and len(og) >= lim:
                        break
                    with SqliteDict(
                        os.path.join(
                            PATH_DATA,
                            "get_foot_type_results.db",
                        ),
                        autocommit=True,
                    ) as db:
                        odx = get_foot_type_results(row.result, db=db)
                    if odx:
                        og.append((i, odx))
                o.extend(og)
            results_df = pd.DataFrame(
                [x[1] for x in o],
                index=[x[0] for x in o],
            )
            odf1 = df.join(results_df)
            return pd.concat(
                [
                    dfshak.assign(
                        author="Shakespeare", model="Shakespeare", num_lines_incl=14
                    ),
                    odf1.assign(author="LLM"),
                ]
            )

    @classmethod
    @cache
    def gather_parse_shakespeare(self, force=False):
        path = os.path.join(PATH_DATA, "gather_parse_shakespeare.json")
        if not force and os.path.exists(path):
            return pd.read_json(path)

        with logmap("gathering and parsing shakespeare") as lm:
            odf = pd.DataFrame(
                {
                    "sonnet_num": stanza.num,
                    "sonnet_fline": stanza.lines[0].txt,
                    **get_foot_type_results(stanza.txt),
                }
                for stanza in lm.iter_progress(
                    prosodic.Text(
                        fn=os.path.join(PATH_DATA, "shakespeare-sonnets.txt")
                    ).stanzas
                )
            )
            odf.to_json(path, orient="records")
            return odf


@cache_obj_rhyme.memoize()
def parse_rhyme_txt(txt, max_rhyme_dist=0):
    import prosodic

    text = prosodic.Text(txt)
    return {
        "num_stanzas": len(text.stanzas),
        "num_lines": len(text.lines),
        "num_rhymes": len(text.get_rhyming_lines(max_dist=max_rhyme_dist)),
        "is_rhyming": text.is_rhyming,
    }


def get_foot_type_results(input_txt, force=False, db=None):

    if not force and input_txt in db:
        return db[input_txt]

    txt = input_txt.strip()
    if txt.startswith("** Sonnet **\n"):
        txt = txt[txt.index("\n") + 1 :]

    with logmap(announce=False) as lm:
        try:
            with lm.verbosity(0):
                poem = prosodic.Text(txt)
                parses = [
                    l.best_parse for l in poem.lines if len(l.best_parse.slots) >= 4
                ][:14]
            # lm.log(len(parses))
            if len(parses) != 14:
                lm.warning(f"parses wrong len {len(parses)}")
                return {}

            if not parses:
                lm.error(f"no parses for {txt}")
                return {}

            perc1 = np.mean([int(parse.slots[3].meter_val == "s") for parse in parses])
            perc2 = np.mean(
                [
                    int(pos.meter_str == "ww")
                    for parse in parses
                    for pos in parse.positions
                ]
            )
            if np.isnan(perc1) or np.isnan(perc2):
                lm.error(f"nan in {perc1}, {perc2} for {txt}")
                return {}
            odx = {
                "meter_perc_lines_fourthpos_s": perc1,
                "meter_mpos_ww": perc2,
                "num_lines_incl": len(parses),
            }
            # lm.log(f"returning {odx}")
            db[input_txt] = odx
            return odx
        except Exception as e:
            lm.error(e)
            return {}
