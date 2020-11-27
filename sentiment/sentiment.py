import os

import pandas as pd

from afinn import Afinn
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_SENTIMENT_SCORES = ["ANEW_Valence", "ANEW_Dominance", "ANEW_Arousal",
                     "Afinn_Valence", "OpinionFinder",
                     "NRC_Arousal", "NRC_Dominance", "NRC_Valence",
                     "NRC_anger", "NRC_anticipation", "NRC_disgust",
                     "NRC_fear", "NRC_joy", "NRC_negative", "NRC_positive",
                     "NRC_sadness", "NRC_surprise", "NRC_trust",
                     "GPOMS_composed_anxious", "GPOMS_agreeable_hostile",
                     "GPOMS_elated_depressed", "GPOMS_confident_unsure",
                     "GPOMS_clearheaded_confused", "GPOMS_energetic_tired",
                     "VADER", "VADER_OpinionFinder",
                     "VADER_Arousal", "VADER_Dominance", "VADER_Valence"]


class Sentiment(object):

    def __init__(self, language="en"):
        """Loads all required lexicons and sentiment scores.

        Since the GPOMS lexicon is not open-source, it is not contained in general version of this package.
        """
        self.language = language
        self.NRC_emotion = self.load_NRC_emotion()
        self.NRC_VAD = self.load_NRC_VAD()
        if language == "en":
            self.afinn = self.load_afinn()
            self.anew = self.load_anew(sort="Mean")
            self.anew_std = self.load_anew(sort="SD")
            self.hedonometer = self.load_hedonometer(sort="Mean")
            self.hedonometer_std = self.load_hedonometer(sort="SD")
            self.opinion_finder = self.load_opinion_finder()
            self.vader_scorer = self.load_vader()
            try:
                self.poms = self.load_poms()
            except FileNotFoundError:
                self.poms = None
        elif language == "nl":
            self.anew = self.load_anew(sort="Mean")
            self.anew_std = self.load_anew(sort="SD")

    def load_NRC_VAD(self):
        """Loads the NRC VAD Lexicon [1].

        Annotations at WORD LEVEL (file: NRC-VAD-Lexicon.txt)

        Returns
        -------
        NRC Emotion lexicon : pandas.DataFrame
            DataFrame containing NRC VAD lexicon and scores

        [1] Saif M. Mohammad. (2018) Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words.
            In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, Melbourne, Australia, July 2018.
        """
        NRC_VAD = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/NRC-VAD-Lexicon.txt"),
                              sep="\t", index_col="Word", na_values="", keep_default_na=False)

        if self.language != "en":
            try:
                trln = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/NRC_translations/{lang}.txt".format(lang=self.language)),
                                   sep="\t", index_col="Word", na_values="", keep_default_na=False)
                NRC_VAD["Word"] = trln.loc[NRC_VAD.index, self.language]
                NRC_VAD.set_index("Word", inplace=True)
            except OSError:
                raise(NotImplementedError)
        return NRC_VAD

    def load_NRC_emotion(self):
        """Loads the NRC Emotion Lexicon [1, 2].

        Annotations at WORD LEVEL (file: NRC-Emotion-Lexicon-Wordlevel-v0.92.txt)

        Returns
        -------
        NRC Emotion lexicon : pandas.DataFrame
            DataFrame containing NRC Emotion lexicon and scores

        [1] Saif Mohammad and Peter Turney (2013), "Crowdsourcing a Word-Emotion Association Lexicon",
            Computational Intelligence, 29 (3), 436-465, 2013.

        [2] Saif Mohammad and Peter Turney (2010), "Emotions Evoked by Common Words and Phrases: Using Mechanical Turk to Create an Emotion Lexicon",
            In Proceedings of the NAACL-HLT 2010 Workshop on Computational Approaches to Analysis and Generation of Emotion in Text, June 2010, LA, California.
        """
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"),
                         sep="\t", header=None, na_values="", keep_default_na=False)
        df.columns = ["word", "category", "score"]
        NRC = df.set_index(["word", "category"]).copy().unstack()
        NRC.columns = NRC.columns.droplevel(0)

        if self.language != "en":
            trln = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/NRC_translations/{lang}.txt".format(lang=self.language)),
                               sep="\t", index_col="Word", na_values="", keep_default_na=False)
            NRC["word"] = trln.loc[NRC.index, self.language]
            NRC.set_index("word", inplace=True)
        return NRC

    def load_afinn(self):
        """Loads the Afinn [1] Valence lexicon and sentiment scores.

        Returns
        -------
        Afinn Valence scores : pandas.DataFrame
            DataFrame containing Afinn Valence lexicon and scores

        [1] Finn Årup Nielsen (2011), "A new ANEW: evaluation of a word list for sentiment analysis in microblogs",
            Proceedings of the ESWC2011 Workshop on 'Making Sense of Microposts': Big things come in small packages.
            Volume 718 in CEUR Workshop Proceedings: 93-98. 2011 May. Matthew Rowe, Milan Stankovic, Aba-Sah Dadzie, Mariann Hardey (editors)
        """
        afinn = pd.DataFrame.from_dict(Afinn(language="en", emoticons=True)._dict, orient="index", columns=["Afinn_Valence"])
        afinn.index.rename("Word", inplace=True)
        return afinn

    def load_vader(self):
        """Loads the VADER [1] lexicon and sentiment analyzer.

        Returns
        -------
        VADER sentiment analyzers : dictionary
            dictionary with several versions of the VADER sentiment analyzer based on different lexicons,
            i.e., original, ANEW Valence, ANEW Dominance, ANEW Arousal, and OpinionFinder

        [1] Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
            Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
        """
        vader = SentimentIntensityAnalyzer()
        VADER = pd.DataFrame(index=pd.Index(vader.lexicon.keys(), name="Word"))
        VADER["VADER"] = pd.Series(vader.lexicon)
        self.vader = VADER
        return {
            "VADER_Valence": SentimentIntensityAnalyzer(lexicon_file=os.path.join(os.path.dirname(__file__), "data/Anew_valence.txt")),
            "VADER_Arousal": SentimentIntensityAnalyzer(lexicon_file=os.path.join(os.path.dirname(__file__), "data/Anew_arousal.txt")),
            "VADER_Dominance": SentimentIntensityAnalyzer(lexicon_file=os.path.join(os.path.dirname(__file__), "data/Anew_dominance.txt")),
            "VADER_OpinionFinder": SentimentIntensityAnalyzer(lexicon_file=os.path.join(os.path.dirname(__file__), "data/Opfi-Sent.txt")),
            "VADER": SentimentIntensityAnalyzer(),
        }

    def load_poms(self):
        """Loads the GPOMS [1] lexicon and sentiment scores.

        Returns
        -------
        GPOMS scores : pandas.DataFrame
            DataFrame containing GPOMS lexicon and scores

        [1] Bollen, J., Mao, H., & Zeng, X.-J. (2011). Twitter Mood Predicts The Stock Market | MIT Technology Review.
        """
        
        poms = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/POMS.csv"), sep='\t', index_col='Word')
        return poms

    def load_hedonometer(self, sort="Mean"):
        """Loads the Hedonometer [1] lexicon and sentiment scores.

        Returns
        -------
        GPOMS scores : pandas.DataFrame
            DataFrame containing Hedonometer lexicon and scores

        [1] Dodds PS, Harris KD, Kloumann IM, Bliss CA, Danforth CM (2011) Temporal Patterns of
            Happiness and Information in a Global Social Network: Hedonometrics and Twitter. PLOS ONE 6(12): e26752.
        """
        if sort not in ["Mean", "SD"]:
            raise(NotImplementedError)

        hedo = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/hedonometer.csv"), sep='\t', index_col='Word')
        if sort == "Mean":
            hedo = hedo[["happiness_average"]]
        elif sort == "SD":
            hedo = hedo[["happiness_standard_deviation"]]

        hedo.columns = ["Happiness"]
        return hedo

    def load_anew(self, sort="Mean"):
        """Loads the CRR ANEW English [1] or Dutch [2] lexicon and sentiment scores.

        Parameters
        ----------
        sort : string
            type of ANEW scores that have to be returned (Mean for mean ANEW values, SD for standard deviation of ANEW values)

        Returns
        -------
        ANEW scores : pandas.DataFrame
            DataFrame containing ANEW lexicon and scores

        [1] Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas.
            Behavior Research Methods, 45, 1191-1207. (http://crr.ugent.be/archives/1003)
        [2] Moors A, De Houwer J, Hermans D, Wanmaker S, van Schie K, Van Harmelen AL, De Schryver M, De Winne J, Brysbaert M. (2013). Norms of valence, arousal, dominance, and age of acquisition for 4,300 Dutch words.
            Behavior Research Methods, 45, 169-77. (http://crr.ugent.be/archives/878)
        """
        if sort not in ["Mean", "SD"]:
            raise(NotImplementedError)

        if self.language == "en":
            anew = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/Ratings_Warriner_et_al.csv"), index_col="Word")
            cols = ['{t}.{s}.Sum'.format(t=t, s=sort) for t in ["V", "A", "D"]]
            anew = anew[cols]
        elif self.language == "nl":
            anew = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/4299WordNorms_Moors_et_al.csv"), index_col=[0], header=[0, 1])
            s = "M" if sort == "Mean" else sort
            cols = [("All", "{s} {t}".format(t=t, s=s)) for t in ["V", "A", "P"]]
            anew = anew[cols]
        anew.columns = ["Valence", "Arousal", "Dominance"]
        return anew

    def load_opinion_finder(self):
        """Loads the OpinionFinder [1] lexicon and sentiment scores.

        Returns
        -------
        OpinionFinder scores : pandas.DataFrame
            DataFrame containing OpinionFinder lexicon and scores

        [1] Wilson, T. & Hoffmann, P. & Somasundaran, S. & Kessler, J. & Wiebe, J. & Choi, Y. & Cardie, C. & Riloff, E. & Patwardhan, S. (2005).
            OpinionFinder: A System for Subjectivity Analysis. HLT/EMNLP 2005: 2005. 10.3115/1225733.1225751. (http://mpqa.cs.pitt.edu/opinionfinder/)
        """
        OpFi = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/OpFi-Sent.txt"), sep="\t", header=None)
        OpFi.columns = ["Word", "OpinionFinder"]
        OpFi.set_index("Word", inplace=True)
        return OpFi

    def calculate_score(self, text, return_all=False, words=[]):
        if return_all:
            return self.calculate_all_scores(text, words=words)
        else:
            return self.calculate_average_score(text, words=words)

    def calculate_all_scores(self, text, words=[]):
        """Scores a text using several sentiment scores.

        Parameters
        ----------
        text : string
            text that has to be processed
        words : list of strings
            words that have to be processed

        Returns
        -------
        sentiment_results : dictionary of score dataframes
            All words that have a sentiment score for all scored sentiments based on text
        """
        if words == []:
            words = tokenize_text(text)

        scores = {}
        scores["NRC_emotion"] = self.score_NRC_emotion(words, return_all=True)
        scores["NRC_VAD"] = self.score_NRC_VAD(words, return_all=True)

        if self.language in ["nl", "en"]:
            scores["anew"] = self.score_anew(words, return_all=True)

        if self.language == "en":
            scores["vader"] = self.score_vader(text, return_all=True)
            scores["afinn"] = self.score_afinn(words, return_all=True)
            scores["opinionfinder"] = self.score_opinion_finder(words, return_all=True)
            scores["happiness"] = self.score_hedonometer(words, return_all=True)
            if self.poms is not None:
                scores["poms"] = self.score_poms(words, return_all=True)
        return scores

    def calculate_average_score(self, text, words=[]):
        """Scores a text using several sentiment scores.

        Parameters
        ----------
        text : string
            text that has to be processed

        Returns
        -------
        sentiment_results : dictionary of scores
            Average sentiment score for all scored sentiments based on text
        """
        if words == []:
            words = tokenize_text(text)

        if self.language == "en":
            sentiment_results = self.score_vader(text)
            afinn_results = self.score_afinn(words)
            if afinn_results:
                sentiment_results["Afinn_Valence"] = afinn_results["Afinn_Valence"]
            else:
                sentiment_results["Afinn_Valence"] = pd.np.nan

            opinion_finder_results = self.score_opinion_finder(words)
            if opinion_finder_results:
                sentiment_results["OpinionFinder"] = opinion_finder_results["OpinionFinder"]
            else:
                sentiment_results["OpinionFinder"] = pd.np.nan

            hedonometer_results = self.score_hedonometer(words)
            if hedonometer_results:
                sentiment_results["Happiness"] = hedonometer_results["Happiness"]
            else:
                sentiment_results["Happiness"] = pd.np.nan

            if self.poms is not None:
                poms_results = self.score_poms(words)
                if poms_results:
                    for sent in poms_results:
                        sentiment_results["GPOMS_" + sent] = poms_results[sent]
                else:
                    for sent in poms_results:
                        sentiment_results["GPOMS_" + sent] = pd.np.nan
        else:
            sentiment_results = {}

        NRC_VAD_results = self.score_NRC_VAD(words)
        if NRC_VAD_results:
            for sent in NRC_VAD_results:
                sentiment_results["NRC_" + sent] = NRC_VAD_results[sent]
        else:
            for sent in NRC_VAD_results:
                sentiment_results["NRC_" + sent] = pd.np.nan

        NRC_results = self.score_NRC_emotion(words)
        if NRC_results:
            for sent in NRC_results:
                sentiment_results["NRC_" + sent] = NRC_results[sent]
        else:
            for sent in NRC_results:
                sentiment_results["NRC_" + sent] = pd.np.nan

        if self.language in ["nl", "en"]:
            anew_results = self.score_anew(words)
            if anew_results:
                for sent in anew_results:
                    sentiment_results["ANEW_" + sent] = anew_results[sent]
            else:
                for sent in anew_results:
                    sentiment_results["ANEW_" + sent] = pd.np.nan

        return sentiment_results

    def score_vader(self, text, return_all=False, dim="compound"):
        """Scores a text using VADER [1].

        Parameters
        ----------
        text : str
            text that has to be processed
        return_all: boolean (optional)
            sets wether to output all word scores (True) or just the mean (False).
            Only uses the complete VADER sentiment scorer if set to True, if set to False
            it will perform a dictionary matching based on a tokenized text.

        Returns
        -------
        scores : pandas.DataFrame of scores (return_all=True) or dictionary of average score (return_all=False)
            VADER sentiment scores for text

        [1] Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
            Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
        """
        if return_all:
            return self.score_dataframe(self.vader, tokenize_text(text), return_all)
        else:
            results = {}
            for sent in self.vader_scorer:
                results[sent] = self.vader_scorer[sent].polarity_scores(text)[dim]
            return results

    def score_afinn(self, words, return_all=False):
        """Scores a list of words using Afinn [1] Valence.

        Parameters
        ----------
        words : iterable of str
            words to be processed
        return_all: boolean (optional)
            sets wether to output all word scores (True) or just the mean (False)

        Returns
        -------
        scores : pandas.DataFrame of scores (return_all=True) or dictionary of average score (return_all=False)
            Afinn Valence scores for words

        [1] Finn Årup Nielsen, "A new ANEW: evaluation of a word list for sentiment analysis in microblogs",
            Proceedings of the ESWC2011 Workshop on 'Making Sense of Microposts': Big things come in small packages.
            Volume 718 in CEUR Workshop Proceedings: 93-98. 2011 May. Matthew Rowe, Milan Stankovic, Aba-Sah Dadzie, Mariann Hardey (editors)
        """
        return self.score_dataframe(self.afinn, words, return_all)

    def score_poms(self, words, return_all=False):
        """Scores a list of words using GPOMS [1].

        Parameters
        ----------
        words : iterable of str
            words to be processed
        return_all: boolean (optional, default False)
            sets whether to output all word scores (True) or just the mean (False)

        Returns
        -------
        scores : pandas.DataFrame of scores (return_all=True) or dictionary of average score (return_all=False)
            POMS scores for words

        [1] Bollen, J., Mao, H., & Zeng, X.-J. (2011). Twitter Mood Predicts The Stock Market | MIT Technology Review.
        """
        return self.score_dataframe(self.poms, words, return_all)

    def score_opinion_finder(self, words, return_all=False):
        """Scores a list of words using OpinionFinder [1].

        Parameters
        ----------
        words : iterable of str
            words to be processed
        return_all: boolean (optional, default False)
            sets whether to output all word scores (True) or just the mean (False)

        Returns
        -------
        scores : pd.DataFrame of scores (return_all=True) or dictionary of average score (return_all=False)
            OpinionFinder scores for words

        [1] Wilson, T. & Hoffmann, P. & Somasundaran, S. & Kessler, J. & Wiebe, J. & Choi, Y. & Cardie, C. & Riloff, E. & Patwardhan, S. (2005).
            OpinionFinder: A System for Subjectivity Analysis. HLT/EMNLP 2005: 2005. 10.3115/1225733.1225751. (http://mpqa.cs.pitt.edu/opinionfinder/)
        """
        return self.score_dataframe(self.opinion_finder, words, return_all)

    def score_anew(self, words, return_all=False):
        """Scores a list of words using ANEW [1].

        Parameters
        ----------
        words : iterable of str
            words to be processed
        return_all: boolean (optional, default False)
            sets whether to output all word scores (True) or just the mean (False)

        Returns
        -------
        scores : pandas.DataFrame of scores (return_all=True) or dictionary of average score (return_all=False)
            ANEW scores for words

        [1] Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas.
            Behavior Research Methods, 45, 1191-1207. (http://crr.ugent.be/archives/1003)
        """
        return self.score_dataframe(self.anew, words, return_all)

    def score_NRC_VAD(self, words, return_all=False):
        """Scores a list of words using NRC VAD Lexicon [1].

        Parameters
        ----------
        words : iterable of str
            words to be processed
        return_all: boolean (optional, default False)
            sets whether to output all word scores (True) or just the count (False)

        Returns
        -------
        scores : pandas.DataFrame of scores (return_all=True) or dictionary of counts (return_all=False)
            for each category in NRC VAD Lexicon

        [1] Saif M. Mohammad. (2018) Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words.
            In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, Melbourne, Australia, July 2018.
        """
        matches = []
        for word in words:
            if word in self.NRC_VAD.index:
                matches.append(word)

        scores = self.NRC_VAD.loc[matches, :]
        if not return_all:
            return scores.sum(axis=0).to_dict()
        else:
            return scores

    def score_NRC_emotion(self, words, return_all=False):
        """Scores a list of words using NRC Emotion Lexicon [1, 2].

        Parameters
        ----------
        words : iterable of str
            words to be processed
        return_all: boolean (optional, default False)
            sets whether to output all word scores (True) or just the count (False)

        Returns
        -------
        scores : pandas.DataFrame of scores (return_all=True) or dictionary of counts (return_all=False)
            for each category in NRC Emotion Lexicon

        [1] Saif Mohammad and Peter Turney (2013), "Crowdsourcing a Word-Emotion Association Lexicon",
            Computational Intelligence, 29 (3), 436-465, 2013.

        [2] Saif Mohammad and Peter Turney (2010), "Emotions Evoked by Common Words and Phrases: Using Mechanical Turk to Create an Emotion Lexicon",
            In Proceedings of the NAACL-HLT 2010 Workshop on Computational Approaches to Analysis and Generation of Emotion in Text, June 2010, LA, California.
        """
        matches = []
        for word in words:
            if word in self.NRC_emotion.index:
                matches.append(word)

        scores = self.NRC_emotion.loc[matches, :]
        if not return_all:
            return scores.sum(axis=0).to_dict()
        else:
            return scores

    def score_hedonometer(self, words, return_all=False):
        """Scores a list of words using Hedonometer [1].

        Parameters
        ----------
        words : iterable of str
            words to be processed
        return_all: boolean (optional, default False)
            sets whether to output all word scores (True) or just the mean (False)

        Returns
        -------
        scores : pandas.DataFrame of scores (return_all=True) or dictionary of average score (return_all=False)
            Hedonometer scores for words

        [1] Dodds PS, Harris KD, Kloumann IM, Bliss CA, Danforth CM (2011) Temporal Patterns of
            Happiness and Information in a Global Social Network: Hedonometrics and Twitter. PLOS ONE 6(12): e26752.
        """
        return self.score_dataframe(self.hedonometer, words, return_all)

    def score_anew_std(self, words):
        """Scores a list of words using CRR ANEW [1].

        Parameters
        ----------
        words : iterable of str
            words to be processed

        Returns
        -------
        scores : pandas.DataFrame of scores (return_all=True) or dictionary of average score (return_all=False)
            ANEW SD values for words

        [1] Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas.
            Behavior Research Methods, 45, 1191-1207. (http://crr.ugent.be/archives/1003)
        """
        return self.score_dataframe(self.anew_std, words, False)

    def score_dataframe(self, df, words, return_all):
        """Scores a list of words using the input dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe of words (index) vs scores (columns)
        words : iterable of str
            words to be processed
        return_all: boolean
            sets whether to output all word scores (True) or just the mean (False)

        Returns
        -------
        scores : pandas.DataFrame of scores (return_all=True) or dictionary of average score (return_all=False)
            Sentiment scores for words based on input DataFrame
        """
        matches = []
        for word in words:
            if word in df.index:
                matches.append(word)

        scores = df.loc[matches, :]
        if not return_all:
            return scores.mean(axis=0).to_dict()
        else:
            return scores


def tokenize_text(text):
    """Tokenizes a given text

    Parameters
    ----------
    text : string
        The text that has to be tokenized

    Returns
    -------
    words : list of tokens
    """
    words = nltk.word_tokenize(text)
    return words
