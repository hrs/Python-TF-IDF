from tfidf import TfIdf
import unittest


class TestTfIdf(unittest.TestCase):
    """For explanation refer to
        https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/"""
    def test_similarity(self):
        table = TfIdf()
        table.add_document("doc1", ["The","game","of","life","is","a","game","of","everlasting","learning"])
        table.add_document("doc2", ["The","unexamined","life","is","not","worth","living"])
        table.add_document("doc3", ["Never","stop","learning"])

        table.calculate_tf()
        table.calculate_idf()
        table.calculate_tf_idf()
        """self.assertEqual(
            table.similarities(["life","learning"]),
            [["foo", 1.0], ["bar", 0.707106781], ["baz", 0.707106781]])"""
        
        print (table.similarities(["life","learning"]))

if __name__ == "__main__":
    unittest.main()
