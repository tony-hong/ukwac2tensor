# conll2xml.py
__author__ = "xhong@coli.uni-sb.com"


import os
import sys
import re
import gzip
import operator
import argparse
import cPickle
from collections import OrderedDict as od
from itertools import izip_longest
import xml.etree.cElementTree as ET
import xml.dom.minidom as mdom

import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn


# Configuration of environment
SRC_DIR = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
sys.path.append(SRC_DIR)

wnl = WordNetLemmatizer()



class CoNLLTools:
    """
    Class that represents a collection of static methods required to parse
    ukWaC corpus.
    <Class implementation in order to wrap a mess of various string processing
    functions into a single class>
    """
    @staticmethod
    def is_ascii(s):
        return all(ord(c) < 128 for c in s)


    @staticmethod
    def validate(token):
        '''validate whether the token is a valid word
        '''
        if re.search(r"[^a-zA-Z']+", token):
            return False
        return True


    @staticmethod
    def penn2wn(pos):
        '''Converts P.O.S. tag from Penn TreeBank style to WordNet style
        '''
        first = pos[0]
        if first == 'J':
            return wn.ADJ
        elif first == 'N':
            return wn.NOUN
        elif first == 'R':
            return wn.ADV
        elif first == 'V':
            return wn.VERB
        return wn.NOUN


    @staticmethod
    def gzip_reader(fname):
        """
        Read a .gz archive and return its contents as a string.
        If the file specified is not an archive it attempt to read it
        as a general file.

        Args:
            *fname* (str) -- file path

        Returns:
            *fdata* (str) -- file contents

        """
        try:
            with gzip.open(fname, 'r') as f:
                fdata = f.read()
        except (OSError, IOError):
            with open(fname, 'r') as f:
                fdata = f.read()
        # return clean_bad_chars(fdata)
        return fdata


    def extract_dataframe(self, conll_data):
        """
        Extract columns from conll files, create an ordered dict of
        ("word", "lemma") pairs and construct sentences for SENNA input.

        Args:
            *data* (str) -- file contents

        Returns:
            | *dataframe* (np.array) -- a dataframe of conll data. 
                Each line is a sentence in CoNLL format:
                    Word    POS    Parse    Predicate   Frameset 1, 2, ...

        """
        data_lines = [cl for cl in conll_data.split('\n')]
        # print (data_lines[8])
        # test wsj
        # assert (data_lines[8].strip() == "")
        
        line = np.array([])
        seentence = np.array([])
        dataframe = []
        first = True
        for dl in data_lines:
            if dl.strip() != "":
                line = np.array(dl.split())
                # if len(line) <= self.COL_2_PROPS: 
                #     continue
                if first:
                    seentence = line
                    first = False
                else:
                    seentence = np.vstack((seentence, line))
            else:
                if len(seentence) != 0:
                    dataframe.append(seentence)
                seentence = np.array([])
                line = np.array([])
                first = True
        # exclude those without Props 
        dataframe = np.array([t for t in dataframe])
        # print (np.max([len(l[0]) for l in dataframe]))
        # print (dataframe[0])

        return dataframe


    def df2malt(self, df):
        new_data = ''
        imgId = 0
        sent_cnt = 0
        isFirst = True

        for line in df:
            # line = sentence[0]
            # print line
            assert line[2] == '_'
            idx = line[0]
            token = line[1]
            pos = line[4]
            dep = line[6]
            syntag = line[7].upper()
            if not self.is_ascii(token):
                token = '<UNKNOWN>'
            lower_token = token.lower()
            lemma = wnl.lemmatize(lower_token, self.penn2wn(pos))
            line[2] = lemma

            new_list = []
            new_list.append(token)
            new_list.append(lemma)
            new_list.append(pos)
            new_list.append(idx)
            new_list.append(dep)
            new_list.append(syntag)
            new_line = '\t'.join(new_list)

            new_data += new_line + '\n'
        
        # print new_sent
        # new_data.append(new_sent)
        # DEBUG block
        # if len(new_data) > 50:
        #     break

        return new_data


    def extract_info(self, dataframe):
        """
        Extract columns from conll files, create an ordered dict of
        ("word", "lemma") pairs and construct sentences for SENNA input.

        Args:
            *data* (str) -- file contents

        Returns:
            | *norm_sents* (str) -- sentences reconstructed from conll and
               separated by "\`\`"
            | *dict_lemmas* (dict) -- a dict of all words and their lemmas
            | *text_ids* (OrderedDict) -- an ordered dict of url string and a
               number of sentences, that belong to this url. Used in order to
               provide source reference for extracted sentences
            | *include_malt_data* (dict) -- dict of sentence number and malt parse
               data extracted from conll file

        """
        # print (dataframe[0])

        result = []
        unique_pairs = {}
        for sentence in dataframe:
            # print sentence
            num_props = len(sentence[0]) - self.COL_2_PROPS
            
            tokens = sentence[:,0]
            
            # lemmas = []
            # for i, t in enumerate(tokens, start=0):
            #     pos = POStags[i]
            #     lemma = wnl.lemmatize(t, CoNLLTools.penn2wn(pos[0]))
            #     lemmas.append(lemma)

            prd_list = predicates[mask]
            # print num_props, len(prd_list)
            assert (num_props == len(prd_list))
            num_tokens = len(tokens)

            ## raw sentence
            # print (' '.join(tokens))
            for prd_id, prd in enumerate(prd_list, start=self.COL_2_PROPS):
                props = sentence[:, prd_id]
                prd_lemma = wnl.lemmatize(prd.lower(), wn.VERB)


                # print (prd, sentence[:, prd_id])
            # print (result)

        plain_output = "\n".join(result)

        print ("Stats:")
        print ("No. of predicates: ", len(unique_pairs.keys()))

        unique_output = ""
        for p, d in unique_pairs.items():
            for w, r in d.items():
                unique_output += str({'V' : p, r : w}) + "\n"

        return plain_output, unique_output

        # dict_lemmas = dict(pairs)
        # sents = ' '.join([w[0] for w in pairs])
        # norm_sents = UkwacTools.insert_sent_delims(UkwacTools.reduce_contr(sents))
        # return norm_sents, dict_lemmas, UkwacTools.include_malt(lines)


    def __init__(self):
        # number of columns until props
        self.COL_2_PROPS = 1
        self.ROLE_LIST = ['A0', 'A1', 'AM-LOC', 'AM-TMP', 'AM-MNR']
        self.ROLE_FILTER = [
            'AM-DIS', 'AM-MOD', 'AM-NEG', 
            'R-A0', 'R-A1', 'R-A2', 
            'R-AM-TMP', 'R-AM-LOC', 'R-AM-CAU', 'R-AM-MNR', 'R-AM-EXT',
            'C-A0', 'C-A1', 'C-V', ]
        # self.ROLE_FILTER = ['AM-MOD', 'AM-NEG', 'AM-DIS']


if __name__ == '__main__':
    prs = argparse.ArgumentParser(description="""
    This script converts conll format data to dat.
    """)
    prs.add_argument('-d', '--dir',
                     help='Specify directory where conll files are located.',
                     required=True)
    prs.add_argument('-p', '--parse',
                     help='Specify conll parsed file to process. ',
                     required=True)
    prs.add_argument('-s', '--srl',
                     help='Specify conll SRL file to process. ', 
                     required=True)
    prs.add_argument('-o', '--out', default=os.getcwd(),
                     help='Specify output directory. If not specified current '
                          'dir is used.',
                     required=False)
    args = prs.parse_args()
    parse_fn = args.parse
    srl_fn = args.srl

    conll_parse_path = os.path.join(args.dir, parse_fn)
    conll_srl_path = os.path.join(args.dir, srl_fn)
    # plain_xml_fname = '.'.join([os.path.basename(filename.strip('.txt')),
    #                         'srl.conll'])
    ct = CoNLLTools()

    print 'reading ...'
    parse_data = ct.gzip_reader(conll_parse_path)
    srl_data = ct.gzip_reader(conll_srl_path)
    print 'extracting dataframe...'
    parse_df = ct.extract_dataframe(parse_data)
    srl_df = ct.extract_dataframe(srl_data)


    print srl_df[1].shape
    print srl_df[1]

    print parse_df[1].shape
    print parse_df[1]

    print ct.df2malt(parse_df[1])
    
    # print 'extracting information...'
    # plain_output, unique_pairs = ct.extract_info(df)

    # print 'creating dat...'
    # result_fname = os.path.join(os.getcwd(), plain_xml_fname)
    # with open(result_fname, 'w+b') as of:
    #     of.write(plain_output)
    # result_fname = os.path.join(os.getcwd(), unique_xml_fname)
    # with open(result_fname, 'w+b') as of:
    #     of.write(unique_pairs)

    # sents, word_lemma, id_index, malt_dic = extract_info(df)
    # build_xml(word_lemma, id_index, result_fname, malt_dic, args.pverbs)
