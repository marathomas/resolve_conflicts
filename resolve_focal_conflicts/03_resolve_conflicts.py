#!/usr/bin/env python
# coding: utf-8

# # Resolve focal conflicts
# 
# Script to resolve conflicts in labeling (e.g. two calls that overlap in time, are highly
# similar and none of them has been labelled nonfocal).
# 
# Requirements:
# - "candidates_matches.json", containing all potential matches (generated with 01_identify_focal_conflicts)
# - "candidates_labelfile.csv", containing audio and spectrogram of all calls involved in a match (generated with 01_identify_focal_conflicts)
# - "f_nf.csv", a csv file containing all pairs of calls and their respective distance scores (generated with 02_assign_distance_score)
# 
# Output:
# - updates f_nf.csv with match scores
# - generates "pred_labelfile.csv", containing the predictions for the candidate calls

# In[1]:


import pandas as pd
import os
import numpy as np
import json
from scipy import stats
import math


# In[2]:


f = open('server_path.txt', "r")
SERVER = f.read().strip()
f.close()

HOME = SERVER + os.path.join(os.path.sep, 'EAS_shared',
                             'meerkat','working','processed',
                             'acoustic', 'resolve_conflicts')

# location of candidate files generated with 01_identify_focal_conflicts
CANDIDATES_MATCHES = os.path.join(os.path.sep, HOME,'candidates_matches.json')
CANDIDATES_LABELFILE = os.path.join(os.path.sep, HOME,'candidates_labelfile.csv')

# location of file with distance scores generated with 02_assign_distance_score
F_NF_FILE = os.path.join(os.path.sep, HOME,'f_nf.csv')


# In[3]:


# Distance score cutoff for deciding which call pairs are same-call vs. different-call
# all <= CUTOFF are labelled as same-call
CUTOFF = 0.25


# In[4]:


# small helper function
def which_call_am_I(call,sub_df):    
    if call==sub_df.call_a.values[0]:
        call="a"
        other="b"
    elif call==sub_df.call_b.values[0]:
        call="b"
        other="a"    
    return call, other


# # Read in data

# In[5]:


print("Reading in data...")


# In[6]:


# labelfile of all candidate matching calls
labelfile = pd.read_csv(CANDIDATES_LABELFILE, sep="\t")
labelfile.shape


# In[7]:


# dictionary of all candidate calls and their potential matches
# (1 call can have multiple matches)
with open(CANDIDATES_MATCHES, "r") as file:  
    matches = json.load(file)
len(matches)


# In[8]:


# table with all call pairs and their respective distance scores
f_nf = pd.read_csv(F_NF_FILE, sep="\t")


# In[19]:


# Dictionary of call audio intensities
intense_dict = dict(zip(f_nf.call_a, f_nf.intense_a))

for call, intense in zip(f_nf.call_b, f_nf.intense_b):
    if call not in intense_dict.keys():
        intense_dict[call] = intense


# # Assign match/no match

# In[ ]:


# previously:
#f_nf['match'] = [1 if x<CUTOFF else 0 for x in f_nf.dist_score]


# When taking GPS constraints into account:

# In[10]:


UPPER_CUTOFF = 0.35
LOWER_CUTOFF = 0.25

UPPER_CUTOFF_GPS = 20
LOWER_CUTOFF_GPS = 2


# In[13]:


match_sim = [1 if ((x<LOWER_CUTOFF) and ~(y>UPPER_CUTOFF_GPS)) else 0 for x,y in zip(f_nf.dist_score, f_nf.physical_dist)]
match_gps = [1 if ((x<UPPER_CUTOFF) and (x>=LOWER_CUTOFF) and (y<LOWER_CUTOFF_GPS)) else 0 for x,y in zip(f_nf.dist_score, f_nf.physical_dist)]
match = [1 if (x==1 or y==1) else 0 for x,y in zip(match_sim, match_gps)]

f_nf['match_sim'] = match_sim
f_nf['match_gps'] = match_gps
f_nf['match'] = match

f_nf.to_csv(F_NF_FILE, sep="\t", index=False)


# # Assign focal or nonfocal

# Make dataframe of all call pairs, their distance scores and faint scores.

# Then assign calls focal or noncal following these hierarchical steps:
# 
# - 1) Assignment of calls with no high fidelity match (e.g. that were not recognized as "same-call")
# - 2) Assignment of calls that have only one high fidelity match
# - 3) Assignment of calls that have multiple high fidelity matches

# In[14]:


print("Assigning focal/nonfocal...")


# In[22]:


# this will save the predictions
pred_nonFocal={}
# this will save information on WHY prediction was made 
why_pred = {}
# this will save additional information on WHY prediction was made 
pred_comment= {}


# ### 1) Case: Calls with no high fidelity match
# 
# --> can be assigned focal (why: "no_match")

# In[23]:


for call in labelfile.callID_new.values:    
    # if not already assigned
    if call not in pred_nonFocal.keys():
        # all rows that concern this call
        sub_df = f_nf.loc[(f_nf['call_a']==call) | (f_nf['call_b']==call),:]
        # all matches of this call        
        sub_df_matched = sub_df.loc[sub_df.match==1,:]
        
        # if call has no matches!:
        if sub_df_matched.shape[0]==0:
            pred_nonFocal[call] = 0
            why_pred[call] = "no_match"
            pred_comment[call] = "no further info"

print(len(pred_nonFocal), " calls assigned after step 1) no high fidelity match...")


# ### 2) Case: Only 1 high fidelity match 

#     3a) the match partner is already assigned f/nf or
#         -->  assign call to the opposite (why: "partner assigned (focal)" or "partner assigned (nonfocal)")
# 
#     3b) match partner is not yet assigned
#          --> assign to nf if weaker, assign to f if stronger (why: "weaker_1_match" or "stronger_1_match")

# In[24]:


for call in labelfile.callID_new.values:    
    # if not already assigned
    if call not in pred_nonFocal.keys():
        # all rows that concern this call
        sub_df = f_nf.loc[(f_nf['call_a']==call) | (f_nf['call_b']==call),:]
        
        # all matches of this call
        sub_df_matched = sub_df.loc[sub_df.match==1,:]
        
        # if call has exactly 1 match:
        if sub_df_matched.shape[0]==1:
            me, other = which_call_am_I(call, sub_df_matched)
            other_call = sub_df_matched['call_'+other].values[0] 
            
            # if partner already assigned
            if other_call in pred_nonFocal.keys():
                # assign to opposite
                pred_nonFocal[call] = np.abs(1-pred_nonFocal[other_call])
                why_pred[call] = "partner_assigned"
                pred_comment[call] = other_call+": "+why_pred[other_call]
                
            # if partner not already assigned   
            else:
                #if weaker, assign nonfocal
                if intense_dict[call]<=intense_dict[other_call]:
                    pred_nonFocal[call] = 1
                    why_pred[call] = "weaker_1_match"
                    pred_comment[call] = str(round(intense_dict[call],2))+" vs. "+str(round(intense_dict[other_call],2))+" ("+other_call+")"
                #if stronger, assign focal
                else:
                    pred_nonFocal[call] = 0
                    why_pred[call] = "stronger_1_match"
                    pred_comment[call] = str(round(intense_dict[call],2))+" vs. "+str(round(intense_dict[other_call],2))+" ("+other_call+")"

print(len(pred_nonFocal), " calls assigned after step 2) ONE high fidelity match...")


# ### 3)  Case: Multiple high fidelity matches
# 
#     4a) of which at least ONE (should be only one) is already known to be the focal one
#         -->  assign call to nonfocal (why: "match with a focal")
# 
#     4b) of which NONE is known to be focal, but
# 
#         4b1) I am the strongest
#             --> assign call to focal (why: "strongest_in_multiple")
# 
#         4b2) I am not the strongest
#             --> assign call to nonfocal (why: "not_strongest_in_multiple")

# In[25]:


for call in labelfile.callID_new.values:    
    # if not already assigned
    if call not in pred_nonFocal.keys():
        # all rows that concern this call
        sub_df = f_nf.loc[(f_nf['call_a']==call) | (f_nf['call_b']==call),:]
        
        # all matches of this call        
        sub_df_matched = sub_df.loc[sub_df.match==1,:]
        
        # if call has  >1 match:
        if sub_df_matched.shape[0]>1:
            all_ids = (list(set(list(sub_df_matched.call_a.values)+list(sub_df_matched.call_b.values))))
            all_partners = [x for x in all_ids if x != call]            
            partner_assignments = [pred_nonFocal[x] for x in all_partners if x in pred_nonFocal.keys()]
            
            #  a) if at least 1 partner is assigned focal
            # (at least one partner is assigned AND at least one is assigned as focal)
            if ((len(partner_assignments)!=0) and (len([x for x in partner_assignments if x==0])>0)):
                pred_nonFocal[call] = 1
                why_pred[call] = "match_with_a_focal"
                
                focal_partner = [x for x in all_partners if ((x in pred_nonFocal.keys()) and (pred_nonFocal[x]==0))]
                pred_comment[call] = focal_partner[0]
            # b) no partner is assigned focal
            else: 
                # b1) I am the strongest
                if intense_dict[call]==np.max([intense_dict[x] for x in all_ids]):
                    pred_nonFocal[call] = 0
                    why_pred[call] = "strongest_in_multiple"
                    pred_comment[call] = str(round(intense_dict[call],2))
                # b2) I am not the strongest
                else:
                    pred_nonFocal[call] = 1
                    why_pred[call] = "not_strongest_in_multiple"
                    pred_comment[call] = str(round(intense_dict[call],2))+" vs."+str(round(np.max([intense_dict[x] for x in all_ids]),2))
                    
                
print(len(pred_nonFocal), " calls assigned after step 3) multiple high fidelity matches")


# # Save results

# In[26]:


labelfile['pred_nonFocal'] = [pred_nonFocal[x] for x in labelfile.callID_new]
labelfile['pred_why'] = [why_pred[x] for x in labelfile.callID_new]
labelfile['pred_comment'] = [pred_comment[x] for x in labelfile.callID_new]


# In[28]:


# can save as csv (if dropping the array columns)
pred_labelfile_out = os.path.join(os.path.sep, HOME,'pred_labelfile.csv')
labelfile.to_csv(pred_labelfile_out, sep="\t", index=False)


# In[29]:


print("Done.")


# # [Check performance]

# In[30]:


print("0: assigned focal, 1: assigned nonfocal")
print(labelfile.pred_nonFocal.value_counts())


# In[15]:


#pd.crosstab(index=labelfile['pred_why'], columns=labelfile['pred_nonFocal'])

