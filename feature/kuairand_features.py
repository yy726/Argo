import pandas as pd

"""
This file contains the feature engineering work for the kuairand 1k dataset.
"""

ACTION_CATEGORY = {
    # 0 is reserved in case we need padding
    "CLICK": 1,
    "LIKE": 2,
    "FOLLOW": 3,
    "COMMENT": 4,
    "FORWARD": 5,
    "HATE": 6,
    "LONG_VIEW": 7,
}

USER_ACTIVE_DEGREE =  ['high_active', 'full_active', 'middle_active', 'UNKNOWN']
FOLLOW_USER_NUM_RANGE = ['0', '(0,10]', '(10,50]', '(100,150]', '(150,250]', '(250,500]', '(50,100]', '500+']
FANS_USER_NUM_RANGE = ['0', '[1,10)', '[10,100)', '[100,1k)', '[1k,5k)', '[5k,1w)', '[1w,10w)']
FRIEND_USER_NUM_RANGE = ['0', '[1,5)', '[5,30)', '[30,60)', '[60,120)', '[120,250)', '250+']
REGISTER_DAYS_RANGE = ['15-30', '31-60', '61-90', '91-180', '181-365', '366-730', '730+']

# the order in this feature list matters as it would affect the value of the category codes
USER_CATEGORY_FEATURES = [
    ("user_active_degree", len(USER_ACTIVE_DEGREE)),
    ("is_lowactive_period", 2),
    ("is_live_streamer", 2),
    ("is_video_author", 2),
    ("follow_user_num_range", len(FOLLOW_USER_NUM_RANGE)),
    ("fans_user_num_range", len(FANS_USER_NUM_RANGE)),
    ("friend_user_num_range", len(FRIEND_USER_NUM_RANGE)),
    ("register_days_range", len(REGISTER_DAYS_RANGE)),
    ("onehot_feat0", 2),
    ("onehot_feat1", 7),
    ("onehot_feat2", 50),
    ("onehot_feat3", 1471),
    ("onehot_feat4", 15),
    ("onehot_feat5", 34),
    ("onehot_feat6", 3),
    ("onehot_feat7", 118),
    ("onehot_feat8", 454),
    ("onehot_feat9", 7),
    ("onehot_feat10", 5),
    ("onehot_feat11", 5),
    ("onehot_feat12", 2),
    ("onehot_feat13", 2),
    ("onehot_feat14", 2),
    ("onehot_feat15", 2),
    ("onehot_feat16", 2),
    ("onehot_feat17", 2),
]

def aggregate_actions(x: pd.Series) -> list[int]:
    actions = []
    if x['is_click'] == 1:
        actions.append(ACTION_CATEGORY["CLICK"])
    if x['is_like'] == 1:
        actions.append(ACTION_CATEGORY["LIKE"])
    if x['is_follow'] == 1:
        actions.append(ACTION_CATEGORY["FOLLOW"])
    if x['is_comment'] == 1:
        actions.append(ACTION_CATEGORY["COMMENT"])
    if x['is_forward'] == 1:
        actions.append(ACTION_CATEGORY["FORWARD"])
    if x['long_view'] == 1:
        actions.append(ACTION_CATEGORY["LONG_VIEW"])
    
    return actions

class KuaiRandFeatureStore:

    def __init__(self, 
                 log: pd.DataFrame, 
                 user_features: pd.DataFrame, 
                 video_features_basic: pd.DataFrame, 
                 video_features_statistic: pd.DataFrame):
        
        pass

    @classmethod
    def extract_user_sequence_features(cls, log: pd.DataFrame) -> pd.DataFrame:
        """
        Extract user sequence features from the log data.

        We would filter out the sequence with at least 1 positive action such as click or like. For
        the item that have multiple positive actions, we would aggregate the positive actions into a
        list.
        """

        # filter out the imp
        filtered_log = log[(log['is_click'] == 1) | (log['is_like'] == 1) | (log['is_follow'] == 1) | (log['is_comment'] == 1) | (log['is_forward'] == 1) | (log['long_view'] == 1)]

        # aggregate the actions over the columns
        filtered_log['actions'] = filtered_log.apply(lambda x: aggregate_actions(x), axis=1)

        # group by user_id and sort by time_ms
        item_sequence = filtered_log.groupby("user_id").apply(lambda x: x.sort_values("time_ms")["video_id"].tolist())
        action_sequence = filtered_log.groupby("user_id").apply(lambda x: x.sort_values("time_ms")["actions"].tolist())

        return item_sequence, action_sequence
    
    @classmethod
    def extract_user_static_features(cls, user_features: pd.DataFrame) -> pd.DataFrame:
        """
        Extract user static features from the user features data.

        The result user features would be converted into a list

        For the categorical features, one solution is to convert them each into a one-hot vector and then concatenate them together.
        Another solution is to convert them into a single category code and use the embedding layer to convert them into a vector.
        For the second solution, we could `merge` them as a single embedding table and treat the original input as multi-hot vector.
        """

        # copy it as a new dataframe to process to make it side effect free
        features = user_features[["user_id"] + ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"] + [x[0] for x in USER_CATEGORY_FEATURES]].copy()
        # extract the numeric features, converted to a single list, which could be further converted to a numpy array
        features["numeric_features"] = features[["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]].apply(lambda x: x.to_numpy(), axis=1)
        features.drop(columns=["follow_user_num", "fans_user_num", "friend_user_num", "register_days"], inplace=True)

        # extract the categorical features and reindex them
        # first encode the string features into integer codes
        features["user_active_degree"] = pd.Categorical(user_features["user_active_degree"], categories=USER_ACTIVE_DEGREE, ordered=True).codes
        features["follow_user_num_range"] = pd.Categorical(user_features["follow_user_num_range"], categories=FOLLOW_USER_NUM_RANGE, ordered=True).codes
        features["fans_user_num_range"] = pd.Categorical(user_features["fans_user_num_range"], categories=FANS_USER_NUM_RANGE, ordered=True).codes
        features["friend_user_num_range"] = pd.Categorical(user_features["friend_user_num_range"], categories=FRIEND_USER_NUM_RANGE, ordered=True).codes
        features["register_days_range"] = pd.Categorical(user_features["register_days_range"], categories=REGISTER_DAYS_RANGE, ordered=True).codes

        # then reindex the categorical features by adding it with the cumulative sum of the pervious features
        cumulative_sum = 0
        for feature, cardinality in USER_CATEGORY_FEATURES:
            features[feature] = features[feature] + cumulative_sum
            cumulative_sum += cardinality
        
        # finally, convert the features into a list
        features["categorical_features"] = features[[x[0] for x in USER_CATEGORY_FEATURES]].apply(lambda x: x.to_numpy(), axis=1)
        features.drop(columns=[x[0] for x in USER_CATEGORY_FEATURES], inplace=True)

        return features
        


