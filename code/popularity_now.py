import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt


def change_pickle_protocol(filepath,protocol=2):
    with open(filepath,'rb') as f:
        obj = pickle.load(f)
    with open(filepath,'wb') as f:
        pickle.dump(obj,f,protocol=protocol)

articles_df = pd.read_pickle('models/articles_df.pkl')


interactions_full_df = pd.read_pickle('models/interactions_full_df.pkl')
interactions_full_indexed_df = pd.read_pickle('models/interactions_full_indexed_df.pkl')
interactions_train_indexed_df = pd.read_pickle('models/interactions_train_indexed_df.pkl')
interactions_test_indexed_df = pd.read_pickle('models/interactions_test_indexed_df.pkl')
interactions_train_df = pd.read_pickle('models/interactions_train_df.pkl')
interactions_test_df = pd.read_pickle('models/interactions_test_df.pkl')





#-----------------------------------------------------------------------------------------
def get_items_interacted(person_id, interactions):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions.loc[person_id]['contentId']
    if(type(interacted_items) == pd.Series):
      return set(interacted_items)
    else:
      return [interacted_items]
    # return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])
#-------------------------------------------------------------------------------------------

#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:


    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        non_interacted_items = set(articles_df['contentId']) - get_items_interacted(person_id, interactions_full_indexed_df)
        # random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items):        
            try:
                # index =999
                listbro= enumerate(recommended_items)
                # for i, c in listbro:
                #   if(c == item_id):
                #     index = next(i)
                index = next(i for i, c in listbro if c == item_id)
            except:
                index = -1
            # for index in range(0,topn):
            #   hit =int(index)
            # hit = int(index in range(0, topn))
            return index

    def evaluate_model_for_user(self, model, person_id):
        #Getting the items in test set
        # interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interactions_test_indexed_df.loc[person_id]['contentId']) == pd.Series:
            person_interacted_items_testset = set(interactions_test_indexed_df.loc[person_id]['contentId'])
        else:
            person_interacted_items_testset = set([int(interactions_test_indexed_df.loc[person_id]['contentId'])])  
        # interacted_items_count_testset = len(person_interacted_items_testset) 

        #Getting a ranked recommendation list from a model for a given user
        temp1=get_items_interacted(person_id,interactions_train_indexed_df)
        temp2=10000000000

       # def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
    #     # Recommend the more popular items that the user hasn't seen yet.
    #     recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(items_to_ignore)] \
    #                            .sort_values('eventStrength', ascending = False) \
    #                            .head(topn)

    #     if verbose:
    #         if self.items_df is None:
    #             raise Exception('"items_df" is required in verbose mode')

    #         recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
    #                                                       left_on = 'contentId', 
    #                                                       right_on = 'contentId')[['eventStrength', 'contentId', 'title', 'url', 'lang']]


    #     return recommendations_df

        recommendations_df = item_popularity_df[~item_popularity_df['contentId'].isin(temp1)] \
                               .sort_values('eventStrength', ascending = False) \
                               .head(temp2)

    #     if verbose:
    #         if self.items_df is None:
    #             raise Exception('"items_df" is required in verbose mode')

        person_recs_df = recommendations_df.merge(articles_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['eventStrength', 'contentId', 'title', 'url', 'lang']]
        
        # person_recs_df = model.recommend_items(person_id,items_to_ignore=temp1, topn=temp2)

        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            #Getting a random sample (100) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)
            temp3=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS
            temp4=item_id%(2**32)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id,sample_size=temp3,seed=temp4)

            #Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample
            items_to_filter_recs=items_to_filter_recs.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            temp5=person_recs_df['contentId'].isin(items_to_filter_recs)
            valid_recs_df = person_recs_df[temp5]                    
            valid_recs = valid_recs_df['contentId']
            valid_recs=valid_recs.values

            index_at_5=self._verify_hit_top_n(item_id, valid_recs)
            hit_at_5 = int(index_at_5 in range(0, 5))
            hits_at_5_count += hit_at_5

            index_at_10=self._verify_hit_top_n(item_id, valid_recs)
            hit_at_10 = int(index_at_10 in range(0, 10))
            hits_at_10_count += hit_at_10

        sizeee= float(len(person_interacted_items_testset))
        recall_at_5 = hits_at_5_count / sizeee
        recall_at_10 = hits_at_10_count / sizeee

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': len(person_interacted_items_testset),
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_model(self, model):
        #print('Running evaluation for users')
        people_metrics = []
        temp6=interactions_test_indexed_df.index.unique()
        temp7=enumerate(list(temp6.values))
        for idx, person_id in temp7:
            #if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        # print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
                            .sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum()
        global_recall_at_5  /= float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum()
        global_recall_at_10 /= float(detailed_results_df['interacted_count'].sum())

        # temp8= model.get_model_name()
        temp8 = 'Popularity'
        
        global_metrics = {'modelName': temp8,
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()


#Computes the most popular items
item_popularity_df = interactions_full_df.groupby('contentId')['eventStrength']
item_popularity_df= item_popularity_df.sum()
item_popularity_df = item_popularity_df.sort_values(ascending=False)
item_popularity_df=item_popularity_df.reset_index()
item_popularity_df.head(10)

class PopularityRecommender:
    
    # MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df
        
    # def get_model_name(self):
    #     return self.MODEL_NAME
        
    # def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
    #     # Recommend the more popular items that the user hasn't seen yet.
    #     recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(items_to_ignore)] \
    #                            .sort_values('eventStrength', ascending = False) \
    #                            .head(topn)

    #     if verbose:
    #         if self.items_df is None:
    #             raise Exception('"items_df" is required in verbose mode')

    #         recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
    #                                                       left_on = 'contentId', 
    #                                                       right_on = 'contentId')[['eventStrength', 'contentId', 'title', 'url', 'lang']]


    #     return recommendations_df
    
popularity_model = PopularityRecommender(item_popularity_df, articles_df)

print('Evaluating Popularity recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
print('\nGlobal metrics:\n%s' % pop_global_metrics)
pop_detailed_results_df.head(10)