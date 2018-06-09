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
from sklearn import preprocessing


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
    #     # similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, self._get_similar_items_to_user_profile(user_id)))
        
    #     recommendations_df = pd.DataFrame(list(filter(lambda x: x[0] not in items_to_ignore,
    #      self._get_similar_items_to_user_profile(user_id)))
    #       , columns=['contentId', 'recStrength']) \
    #                                 .head(topn)

    #     if verbose:
    #     #     if self.items_df is None:
    #     #         raise Exception('"items_df" is required in verbose mode')

    #         recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
    #                                                       left_on = 'contentId', 
    #                                                       right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]


    #     return recommendations_df




    # def _get_similar_items_to_user_profile(self, person_id, topn=1000):
    #     #Computes the cosine similarity between the user profile and all item profiles
    #     temp15=user_profiles[person_id]
    #     cosine_similarities = cosine_similarity(temp15, tfidf_matrix)
    #     #Gets the top similar items
    #     similar_indices = cosine_similarities.argsort()
    #     similar_indices = similar_indices.flatten()
    #     similar_indices = similar_indices[-topn:]
    #     #Sort the similar items by similarity
    #     temp16=[]
    #     for i in similar_indices:
    #       temp17=item_ids[i]
    #       temp18=[]
    #       temp18.append(temp17);
    #       temp18.append(cosine_similarities[0,i])
    #       temp16.append(temp18)

    #     similar_items = sorted(temp16, key=lambda x: -x[1])
    #     return similar_items


        temp15=user_profiles[person_id]
        cosine_similarities = cosine_similarity(temp15, tfidf_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort()
        similar_indices = similar_indices.flatten()
        similar_indices = similar_indices[-1000:]
        #Sort the similar items by similarity
        temp16=[]
        for i in similar_indices:
          temp17=item_ids[i]
          temp18=[]
          temp18.append(temp17);
          temp18.append(cosine_similarities[0,i])
          temp16.append(temp18)

        similar_items = sorted(temp16, key=lambda x: -x[1])


        recommendations_df = pd.DataFrame(list(filter(lambda x: x[0] not in temp1,
         similar_items))
          , columns=['contentId', 'recStrength']) \
                                    .head(temp2)

        person_recs_df = recommendations_df.merge(articles_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]

        
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
        temp8 = 'Content-Based'
        
        global_metrics = {'modelName': temp8,
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()

#------------------------------------------------------------------------------------------------


#Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
stopwords_list = stopwords.words('english') 
stopwords_list += stopwords.words('portuguese')

#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 3),
                     min_df=0.003,
                     max_df=0.1,
                     max_features=5000,
                     norm='l1',
                     stop_words=stopwords_list)
temp9=articles_df['contentId']
item_ids = temp9.tolist()
temp10= articles_df['title'] + "" + articles_df['text']
tfidf_matrix = vectorizer.fit_transform(temp10)
tfidf_feature_names = vectorizer.get_feature_names()

# -------------------------------------------------------------------------------------------------

# def get_item_profile(item_id):
#     # idx = item_ids.index(item_id)
#     item_profile = tfidf_matrix[item_ids.index(item_id) : item_ids.index(item_id) + 1]
#     return item_profile

def get_item_profiles(ids):
    item_profiles_list = [] #[get_item_profile(x) for x in ids]
    for x in ids:
      item_profiles_list.append(tfidf_matrix[item_ids.index(x) : item_ids.index(x) + 1])
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles

def build_users_profile(person_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[person_id]
    temp11= interactions_person_df['contentId']
    temp12= interactions_person_df['eventStrength']
    user_item_profiles = get_item_profiles(temp11)
    
    user_item_strengths = np.array(temp12)
    user_item_strengths = user_item_strengths.reshape(-1,1)
    #Weighted average of item profiles by the interactions strength
    temp13 =user_item_profiles
    temp13 = temp13.multiply(user_item_strengths)
    user_item_strengths_weighted_avg = np.sum(temp13, axis=0)
    user_item_strengths_weighted_avg /= np.sum(user_item_strengths)
    user_profile_norm = preprocessing.normalize(user_item_strengths_weighted_avg)
    return user_profile_norm

def build_users_profiles(): 
    interactions_indexed_df = interactions_full_df[interactions_full_df['contentId'] \
                                                   .isin(articles_df['contentId'])]
    interactions_indexed_df= interactions_indexed_df.set_index('personId')
    temp14= interactions_indexed_df.index.unique()
    user_profiles = {}
    for person_id in temp14:
        user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
    return user_profiles

user_profiles = build_users_profiles()
# print len(user_profiles)

#-----------------------------------------------------------------------------------

class ContentBasedRecommender:
    
    # MODEL_NAME = 'Content-Based'
    
    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df
        
    # def get_model_name(self):
    #     return self.MODEL_NAME
        
    # def _get_similar_items_to_user_profile(self, person_id, topn=1000):
    #     #Computes the cosine similarity between the user profile and all item profiles
    #     temp15=user_profiles[person_id]
    #     cosine_similarities = cosine_similarity(temp15, tfidf_matrix)
    #     #Gets the top similar items
    #     similar_indices = cosine_similarities.argsort()
    #     similar_indices = similar_indices.flatten()
    #     similar_indices = similar_indices[-topn:]
    #     #Sort the similar items by similarity
    #     temp16=[]
    #     for i in similar_indices:
    #       temp17=item_ids[i]
    #       temp18=[]
    #       temp18.append(temp17);
    #       temp18.append(cosine_similarities[0,i])
    #       temp16.append(temp18)

    #     similar_items = sorted(temp16, key=lambda x: -x[1])
    #     return similar_items
        
    # def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
    #     # similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, self._get_similar_items_to_user_profile(user_id)))
        
    #     recommendations_df = pd.DataFrame(list(filter(lambda x: x[0] not in items_to_ignore,
    #      self._get_similar_items_to_user_profile(user_id)))
    #       , columns=['contentId', 'recStrength']) \
    #                                 .head(topn)

    #     if verbose:
    #     #     if self.items_df is None:
    #     #         raise Exception('"items_df" is required in verbose mode')

    #         recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
    #                                                       left_on = 'contentId', 
    #                                                       right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]


    #     return recommendations_df
    
content_based_recommender_model = ContentBasedRecommender(articles_df)

# -----------------------------------------------------------------------------------------

print('Evaluating Content-Based Filtering model...')
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
print('\nGlobal metrics:\n%s' % cb_global_metrics)
cb_detailed_results_df.head(10)