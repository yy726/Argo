import pandas as pd
import unittest

from feature.kuairand_features import KuaiRandFeatureStore


class TestKuaiRandFeatures(unittest.TestCase):

    def test_extract_user_sequence_features(self):
        log = pd.read_csv("test/resources/kuairand-1k-test.csv")
        item_sequence, action_sequence = KuaiRandFeatureStore.extract_user_sequence_features(log)

        assert len(item_sequence.iloc[0]) == 176
        assert len(item_sequence.iloc[1]) == 338

        assert len(action_sequence.iloc[0]) == 176
        assert len(action_sequence.iloc[1]) == 338

        assert item_sequence.iloc[0][0] == 2528540
        assert action_sequence.iloc[0][0] == [1]
        assert item_sequence.iloc[0][1] == 4067506
        assert action_sequence.iloc[0][1] == [1, 7]

        # check the item that is clicked, liked and log view
        for i in range(len(item_sequence.iloc[1])):
            if item_sequence.iloc[1][i] == 4001361:
                assert action_sequence.iloc[1][i] == [1, 2, 7]

        print("KuaiRandFeatures test passed...")

    def test_extract_user_features(self):
        user_features = pd.read_csv("test/resources/user-features-test.csv")

        features = KuaiRandFeatureStore.extract_user_static_features(user_features)

        assert features.shape == (5, 3)
        assert features.iloc[0]["user_id"] == 0
        assert all(features.iloc[0]["numeric_features"] == [514, 150, 34, 799])
        assert all(features.iloc[1]["numeric_features"] == [457, 20, 3, 1474])

        assert all(features.iloc[0]["categorical_features"] == [1, 4, 7, 9, 17, 21, 28, 38, 40, 42, 77, 1047, 1570, 1584, 1618, 1635,
                                                                1874, 2199, 2203, 2205, 2210, 2213, 2214, 2216, 2218, 2220])