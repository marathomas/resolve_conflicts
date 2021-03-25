#!/usr/bin/env python
# coding: utf-8

# # Update labelfiles with predictions

# Script to update labelfile with predictions from resolve_conflicts
# 
# Requirements:
# - "pred_labelfile.csv" containing the predictions for the candidate calls generated with 03_resolve_focal_conflicts
# - all original labelfiles in folder "EAS_shared/meerkat/working/processed/acoustic/total_synched_call_tables/"
# 
# Output:
# - updates the individual labelfile csvs with the predictions for the candidate calls and saves updated with ending "_conflicts_resolved.csv" in current working directory

# In[1]:


import pandas as pd
import os
import numpy as np


# In[2]:


f = open('server_path.txt', "r")
SERVER = f.read().strip()
f.close()

HOME = SERVER + os.path.join(os.path.sep, 'EAS_shared',
                             'meerkat','working','processed',
                             'acoustic', 'resolve_conflicts')

PRED_LABELFILE = os.path.join(os.path.sep, HOME,'pred_labelfile.csv')

# folder containing label csvs indicating start, stop times etc
LABELFILES_FOLDER = SERVER + os.path.join(os.path.sep, 'EAS_shared',
                                         'meerkat','working','processed',
                                         'acoustic', 'total_synched_call_tables')


# In[3]:


def replace_multiple(string, list_of_chars, replacement):
    """
    Function that replaces multiple substrings in a string
    with other substrings.

    Parameters
    ----------
    string : String
             your input string 
    list_of_chars: list of strings
                   List of substrings you want to have replaced
    replacement: string or list of strings
                 Substring or list of substrings you want to use as
                 replacement. If list, then it should be the same length as
                 list_of_chars to be matched by position.

    Returns
    -------
    result : String
             The modified string

    Example
    -------
    >>> mod_string = replace_multiple("This is an example", ['s', 'a'], '!')
    >>> 'Thi! i! !n ex!mple'
    
    >>> mod_string = replace_multiple("This is an example", ['s', 'a'], ['S', 'A'])
    >>> 'ThiS iS An exAmple'
    """ 
    # if all are to be replaced by same string
    if (type(replacement)==str):
        replacement = [replacement]*len(list_of_chars)
        
    for ch, repl in zip(list_of_chars, replacement):
        if ch in string:
            string=string.replace(ch,repl)
    return string


# In[5]:


pred_labelfile = pd.read_csv(PRED_LABELFILE,sep="\t")

pred_nf_dict = dict(zip(pred_labelfile.callID_new, ["F" if x==0 else "NF" for x in pred_labelfile.pred_nonFocal]))
pred_why_dict = dict(zip(pred_labelfile.callID_new, pred_labelfile.pred_why))
pred_comment_dict = dict(zip(pred_labelfile.callID_new, pred_labelfile.pred_comment))


# In[8]:


# Read in all labelfiles 
labelfiles_list = os.listdir(LABELFILES_FOLDER)
print("Updating files...")

for file in labelfiles_list:
    
    df = pd.read_csv(os.path.join(os.path.sep, LABELFILES_FOLDER,file), sep="\t", encoding="ISO-8859-1")
    
    # make new column callID_new without problematic chars
    to_be_replaced = ["/", " ", ":", "."]
    replace_with = "_"
    new_callID = [replace_multiple(x, to_be_replaced, replace_with) for x in df.callID]
    df['callID_new'] = new_callID
    
    # make new column pred_focalType with updated type from resolve_conflicts 
    df['pred_focalType'] = [pred_nf_dict[callID] if callID in pred_nf_dict.keys() else human_nf_label for callID,human_nf_label in zip(df.callID_new, df.focalType)]
    df['pred_focalType_why'] = [pred_why_dict[callID] if callID in pred_why_dict.keys() else "NA" for callID in df.callID_new]
    df['pred_focalType_comment'] = [pred_comment_dict[callID] if callID in pred_comment_dict.keys() else "NA" for callID in df.callID_new]
    
    # save
    filename = file[:-4]
    
    outname = os.path.join(os.path.sep, HOME,filename+'_conflicts_resolved.csv')
    df.to_csv(outname, sep="\t", index=False)
    print(outname)


# In[ ]:


print("Done.")

