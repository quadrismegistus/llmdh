from .imports import *


class LLM:
    model = LLM_DEFAULT_MODEL
    models = [LLM_DEFAULT_MODEL]
    user_prompt = ""
    system_prompt = ""
    example_prompts = []
    filekey = "LLM"
    filename = ""
    input_data = {}

    ## class methods
    @classmethod
    @cache
    def openai_api(cls):
        api_key = os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key, base_url=OPENAI_BASEURL)
        # client.api_key=api_key
        return client

    @classmethod
    @cache
    def claude_api(cls):
        from anthropic import Anthropic

        api_key = os.environ.get("CLAUDE_API_KEY")
        client = Anthropic(api_key=api_key)
        return client

    @classmethod
    @cache
    def gemini_api(cls, model="gemini-pro"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(model)
        return model

    @classmethod
    def format_prompt(
        cls,
        user_prompt: str,
        system_prompt: str = "",
        example_prompts: List[Tuple[str, str]] = [],
        model: str = LLM_DEFAULT_MODEL,
        as_str: bool = False,
        **kwargs,
    ):
        o = []
        if system_prompt:
            o.append({"role": "system", "content": system_prompt})
        if example_prompts:
            for question, answer in example_prompts:
                o.append({"role": "user", "content": question})
                o.append({"role": "assistant", "content": answer})
        o.append({"role": "user", "content": user_prompt})

        if "gemini" in model:
            as_str = True
        if as_str:
            o = "\n\n\n".join(
                "## " + d["role"].upper() + " ##\n\n" + d["content"].strip() for d in o
            )
        return o

    @classmethod
    def generate(
        cls,
        user_prompt: str = "",
        system_prompt: str = "",
        example_prompts: List[Tuple[str, str]] = [],
        model: str = LLM_DEFAULT_MODEL,
        verbose: bool = True,
        max_tokens=MAX_TOKENS,
        name=None,  # optional name for logging errors
        **kwargs,
    ):
        kwargs = dict(
            user_prompt=user_prompt if user_prompt else cls.user_prompt,
            system_prompt=system_prompt if system_prompt else cls.system_prompt,
            example_prompts=example_prompts if example_prompts else cls.example_prompts,
            model=model if model else cls.model,
            verbose=verbose,
            max_tokens=max_tokens,
            name=name,
            **kwargs,
        )
        if "gemini" in model:
            return cls.generate_gemini(**kwargs)
        elif "gpt" in model:
            return cls.generate_openai(**kwargs)
        elif "claude" in model:
            return cls.generate_claude(**kwargs)
        else:
            return cls.generate_ollama(**kwargs)

    @classmethod
    def generate_ollama(
        cls,
        user_prompt,
        *args,
        model="llama2-uncensored:7b",
        verbose=True,
        name=None,
        temp=DEFAULT_TEMP,
        **kwargs,
    ):
        with logmap(f"prompting LLM model {model}", announce=verbose) as lm:
            prompt = cls.format_prompt(user_prompt, *args, model=model, **kwargs)
            if verbose:
                lm.log(f"PROMPT: {prompt}")
            try:
                response = ollama.chat(
                    model=model,
                    messages=prompt,
                    options={"temperature": temp},
                )
                text = response["message"]["content"].strip()
                if verbose:
                    lm.warning(f"RESPONSE: {text}")
                return text
            except Exception as e:
                lm.error(e)
                return ""

    @classmethod
    def generate_openai(
        cls,
        user_prompt,
        *args,
        model=LLM_DEFAULT_MODEL_OPENAI,
        verbose=False,
        max_tokens=MAX_TOKENS,
        name=None,
        temp=DEFAULT_TEMP,
        model_kwargs={},
        **kwargs,
    ):
        res = None
        with logmap(f"prompting LLM model {model}", announce=verbose) as lm:
            prompt = cls.format_prompt(user_prompt, *args, model=model, **kwargs)
            # if verbose: lm.log(f'PROMPT: {prompt}')
            try:
                chat_completion = cls.openai_api().chat.completions.create(
                    messages=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temp,
                    **model_kwargs,
                )
                text = (
                    chat_completion.choices[0].message.content.strip()
                    if chat_completion.choices
                    else ""
                )
                if verbose:
                    lm.warning(f"RESPONSE: {text}")
                return text
            except Exception as e:
                logger.error(e)
                return ""

    @classmethod
    def generate_claude(
        cls,
        user_prompt,
        *args,
        model=LLM_DEFAULT_MODEL_CLAUDE,
        verbose=False,
        max_tokens=MAX_TOKENS,
        name=None,
        temp=DEFAULT_TEMP,
        model_kwargs={},
        **kwargs,
    ):
        res = None
        with logmap(f"prompting LLM model {model}", announce=verbose) as lm:
            prompt = cls.format_prompt(user_prompt, *args, model=model, **kwargs)
            # if verbose: lm.log(f'PROMPT: {prompt}')
            try:
                chat_completion = cls.claude_api().messages.create(
                    messages=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temp,
                    **model_kwargs,
                )
                text = "\n\n".join(result.text for result in chat_completion.content)
                if verbose:
                    lm.warning(f"RESPONSE: {text}")
                return text
            except Exception as e:
                logger.error(e)
                return ""

    @classmethod
    def generate_gemini(
        cls,
        user_prompt,
        *args,
        model="gemini-pro",
        verbose=False,
        name=None,
        temp=DEFAULT_TEMP,
        **kwargs,
    ):
        res = None
        with logmap(f"prompting LLM model {model}", announce=verbose) as lm:
            prompt = cls.format_prompt(user_prompt, *args, model=model, **kwargs)
            if verbose:
                lm.log(f"PROMPT: {prompt}")
            try:
                # res = gemini_api().generate_content(prompt)
                res = cls.gemini_api(model=model).generate_content(
                    prompt,
                    generation_config=GenerationConfig(
                        max_output_tokens=None,
                        temperature=temp,
                    ),
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    },
                    request_options={"timeout": 600},
                )

                text = "\n\n".join(
                    part.text for part in res.parts if hasattr(part, "text")
                )
                if verbose:
                    lm.warning(f"RESPONSE: {text}")
                return text
            except Exception as e:
                logger.error(e)
                return ""

    ## instance methods

    def __init__(
        self,
        user_prompt: str = "",
        system_prompt: str = "",
        example_prompts: List[Tuple[str, str]] = [],
        model=LLM_DEFAULT_MODEL,
        verbose=False,
        max_tokens=MAX_TOKENS,
        name: str = "",
        filename: str = "",
        filekey: str = "",
        force=False,
        **kwargs,
    ):

        ## override class attrs?
        if model:
            self.model = model
        if user_prompt:
            self.user_prompt = user_prompt
        if system_prompt:
            self.system_prompt = system_prompt
        if example_prompts:
            self.example_prompts = example_prompts
        if filekey:
            self.filekey = filekey
        elif not self.filekey:
            self.filekey = self.__class__.__name__
        if filename:
            self.filename = filename
        elif not self.filename:
            self.filename = self.get_filename(filekey=self.filekey, model=model)

        ## instance attrs
        self.name = self.user_prompt if not name else name
        self.prompt_feedback = None
        self.verbose = verbose
        self.max_tokens = max_tokens
        self.article = None
        self._force = force

    def get_prompt(self, as_str=False):
        return self.format_prompt(
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
            example_prompts=self.example_prompts,
            model=self.model,
            as_str=as_str,
        )

    @property
    def prompt(self):
        return self.get_prompt()

    def gen(self, verbose=None, max_tokens=None):
        return type(self).generate(
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
            example_prompts=self.example_prompts,
            model=self.model,
            verbose=self.verbose if verbose is None else verbose,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            name=self.name,
        )

    @cached_property
    def is_valid(self):
        return self.parsed_response is not None

    @classmethod
    @property
    def path_db(self):
        return self.get_path_db()

    @classmethod
    @property
    def db(self):
        return self.get_db()

    @classmethod
    @property
    def db_read(self):
        if not os.path.exists(self.path_db):
            flag = "c"
        else:
            flag = "r"
        return self.get_db(flag=flag)

    @classmethod
    def get_db(self, flag="c", model="", filekey=""):
        return SqliteDict(
            self.get_path_db(model=model, filekey=filekey), autocommit=True, flag=flag
        )

    @classmethod
    def get_filename(self, model="", filekey=""):
        if self.filename:
            return self.filename
        return f"data.{filekey if filekey else self.filekey}.llm.{model if model else self.model}.sqlitedict"

    @classmethod
    def get_path_db(self, model="", filekey=""):
        fn = self.get_filename(model=model, filekey=filekey)
        return os.path.join(PATH_DATA, fn) if not os.path.isabs(fn) else fn

    @property
    def raw(self):
        with self.db_read as db:
            return db.get(self.name)

    @property
    def cached_result(self):
        ld = self.raw
        return ld[-1].get("result") if ld else None

    @cached_property
    def response(self):
        with self.db_read as db:
            if not self._force and self.name in db:
                res = db[self.name][-1]["response"]
                if res:
                    return res
        return self.gen()

    @cached_property
    def result(self):
        res = self.cached_result
        if self._force or not res:
            res = self.parsed_response
            self.save()
        return res

    @cached_property
    def parsed_response(self):
        return self.response

    def save(self):
        with self.get_db() as db:
            if self.is_valid:
                outd = {
                    "model": self.model,
                    "system_prompt": self.system_prompt,
                    "example_prompts": self.example_prompts,
                    "user_prompt": self.user_prompt,
                    "input_data": self.input_data,
                    "response": self.response,
                    "result": self.parsed_response,
                    "is_valid": self.is_valid,
                }
                outkey = self.name
                with self.db as db:
                    if not outkey in db:
                        db[outkey] = [outd]
                    else:
                        try:
                            already = any(
                                (x["result"] == outd["result"]) for x in db[outkey]
                            )
                            if already:
                                logger.warning("already saved")
                                return
                        except Exception:
                            pass
                        db[outkey] = db[outkey] + [outd]

    @classmethod
    def gather(self):
        o = []
        for model in self.models:
            db = self.get_db(model=model)
            for key in sorted(list(db.keys())):
                for d in db[key]:
                    if type(d["result"]) == dict:
                        resd = d.pop("result")
                        odx = {**d, **resd}
                    else:
                        odx = {**d}
                    o.append(odx)
        return pd.DataFrame(o)

    @classmethod
    def run(cls, model=LLM_DEFAULT_MODEL, verbose=False, force=False, **kwargs):
        with logmap(f"running RhymeLLM using model {model}") as lm:
            last_res = None
            while True:
                llm = cls(model=model, verbose=verbose, force=force, **kwargs)
                if last_res == llm.result:
                    lm.warning("already")
                else:
                    if verbose:
                        print(pformat(llm.result))
                last_res = llm.result
                time.sleep(random.random() * 2)

    @classmethod
    @cache
    def calculate_iaa(self, cols, index, passes_keyword_filter=True):
        col2model2series = defaultdict(dict)
        with logmap("calculating iaa") as lm:
            df = self.gather().reset_index()
            if passes_keyword_filter:
                df = df[df.passes_keyword_filter]

            for model, mdf in df.set_index(list(index)).groupby("model"):
                for col in cols:
                    s = mdf[col]
                    s = s[s != ""]
                    s = s[s != None]
                    s = s.apply(maybe_int)
                    col2model2series[col][model] = s

            ntry = 25
            out = []
            for col in lm(col2model2series):
                lm.set_progress_desc("comparing along " + col)
                for mdl1, mdl2 in combos(col2model2series[col]):
                    s1 = col2model2series[col][mdl1].apply(str)
                    s2 = col2model2series[col][mdl2].apply(str)
                    index = list(set(s1.index) & set(s2.index))
                    if not index:
                        continue
                    for i in range(ntry):
                        indexes = random.choices(index, k=50)
                        ss1 = s1.loc[indexes]
                        ss2 = s2.loc[indexes]
                        pos_label = max(
                            list(set(list(ss1.unique()) + list(ss2.unique())))
                        )
                        odx = {
                            "col": col,
                            "model1": mdl1,
                            "model2": mdl2,
                            "num1": len(s1),
                            "num2": len(s2),
                            "run": i,
                            **calculate_iaa_stats(ss1, ss2, pos_label=pos_label),
                        }
                        out.append(odx)

            odf = pd.DataFrame(out)
            return odf.sort_values("cohen_kappa", ascending=False)


#####


def hashstr(*inputs, length=12):
    import hashlib

    input_string = str(inputs)
    sha256_hash = hashlib.sha256(str(input_string).encode()).hexdigest()
    return sha256_hash[:length]
