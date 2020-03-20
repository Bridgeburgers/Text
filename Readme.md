Text Style Translation
======================

This is an attempt to translate a sentence into a new sentence with a different
style. The method is to combine a text generator, learned in a particular style
with an LSTM, with a sentence encoder in the form of a pre-trained transformer
(in this case BERT) to create a utility function for a sentence to match the
content of an input sentence and the style of the generator.

There are three major components to this repository:

1.  Loading and processing a text file into the proper format for learning for
    the text generator (*UrlToText.py*)

2.  The classes and functions for building a word-level text generator
    (*WordGenerator.py*)

3.  The searcher which both defines the utility function for sentence matching,
    and searches for the sentence that maximizes utility in a stochastic manner.
    In this instance, using a version of single-player monte carlo tree search.
    (*PhraseSearch.py*)

The file *WordGeneratorTemplate.py* shows how to use these modules to load a
text file, train (and save) a word generator, and start a search.

 

As of this time, the searcher is not very successful. But, at the very least,
this repository can be used to create a text generator for any input corpus.
